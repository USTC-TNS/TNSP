/**
 * \file sampling_gradient_lattice.hpp
 *
 * Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once
#ifndef SQUARE_SAMPLING_GRADIENT_LATTICE_HPP
#define SQUARE_SAMPLING_GRADIENT_LATTICE_HPP

#include "abstract_network_lattice.hpp"
#include "square_auxiliaries_system.hpp"

namespace square {
   template<typename T>
   struct SpinConfiguration : SquareAuxiliariesSystem<T> {
      const SamplingGradientLattice<T>* owner;
      std::vector<std::vector<int>> configuration;

      using SquareAuxiliariesSystem<T>::M;
      using SquareAuxiliariesSystem<T>::N;
      using SquareAuxiliariesSystem<T>::dimension_cut;
      using SquareAuxiliariesSystem<T>::lattice;
      using SquareAuxiliariesSystem<T>::operator();

      SpinConfiguration() = default;

      SpinConfiguration(const SamplingGradientLattice<T>* owner) :
            SquareAuxiliariesSystem<T>(owner->M, owner->N, owner->dimension_cut), owner(owner) {
         for (auto i = 0; i < M; i++) {
            auto& row = configuration.emplace_back();
            for (auto j = 0; j < N; j++) {
               row.push_back(-1);
            }
         }
      }

      void set(const std::tuple<int, int>& position, int spin) {
         auto [x, y] = position;
         if (configuration[x][y] != spin) {
            if (spin == -1) {
               lattice[x][y]->unset();
               configuration[x][y] = spin;
            } else {
               lattice[x][y]->set(owner->lattice[x][y].shrink({{"P", spin}}));
               configuration[x][y] = spin;
            }
         }
      }

      auto operator()(const std::map<std::tuple<int, int>, int>& replacement) const {
         auto real_replacement = std::map<std::tuple<int, int>, Tensor<T>>();
         for (auto& [position, spin] : replacement) {
            auto [x, y] = position;
            if (configuration[x][y] != spin) {
               real_replacement[{x, y}] = owner->lattice[x][y].shrink({{"P", spin}});
            }
         }
         return operator()(real_replacement);
      }

      auto operator()() const {
         return operator()(std::map<std::tuple<int, int>, Tensor<T>>());
      }
   };

   template<typename T>
   struct SamplingGradientLattice : AbstractNetworkLattice<T> {
      Size dimension_cut;
      SpinConfiguration<T> spin;

      // spin应当只用this初始化, 随后initialize_spin即可
      SamplingGradientLattice() = default;

      SamplingGradientLattice(const SamplingGradientLattice<T>& other) :
            AbstractNetworkLattice<T>(other), dimension_cut(other.dimension_cut), spin(this) {
         initialize_spin(other.spin.configuration);
      }
      SamplingGradientLattice(SamplingGradientLattice<T>&& other) :
            AbstractNetworkLattice<T>(std::move(other)), dimension_cut(other.dimension_cut), spin(this) {
         initialize_spin(other.spin.configuration);
      }
      SamplingGradientLattice<T>& operator=(const SamplingGradientLattice<T>& other) {
         if (this != &other) {
            new (this) SamplingGradientLattice<T>(other);
         }
         return *this;
      }
      SamplingGradientLattice<T>& operator=(SamplingGradientLattice<T>&& other) {
         if (this != &other) {
            new (this) SamplingGradientLattice<T>(std::move(other));
         }
         return *this;
      }

      SamplingGradientLattice(int M, int N, Size D, Size Dc, Size d) : AbstractNetworkLattice<T>(M, N, D, d), dimension_cut(Dc), spin(this) {}

      explicit SamplingGradientLattice(const SimpleUpdateLattice<T>& other, Size Dc);

      using AbstractNetworkLattice<T>::M;
      using AbstractNetworkLattice<T>::N;
      using AbstractNetworkLattice<T>::dimension_physics;
      using AbstractNetworkLattice<T>::hamiltonians;
      using AbstractNetworkLattice<T>::dimension_virtual;
      using AbstractNetworkLattice<T>::lattice;

      void initialize_spin(std::function<int(int, int)> function) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               spin.set({i, j}, function(i, j));
            }
         }
      }

      void initialize_spin(const std::vector<std::vector<int>>& configuration) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               spin.set({i, j}, configuration[i][j]);
            }
         }
      }

      auto markov(
            unsigned long long total_step,
            std::map<std::string, std::map<std::vector<std::tuple<int, int>>, std::shared_ptr<const Tensor<T>>>> observers,
            bool calculate_energy = false) {
         std::cout << clear_line << "Markov sampling start, total_step=" << total_step << ", dimension=" << dimension_virtual
                   << ", dimension_cut=" << dimension_cut << "\n"
                   << std::flush;
         if (calculate_energy) {
            observers["Energy"] = hamiltonians;
         }
         auto result = std::map<std::string, std::map<std::vector<std::tuple<int, int>>, real<T>>>();
         auto result_variance_square = std::map<std::string, std::map<std::vector<std::tuple<int, int>>, real<T>>>();
         auto result_square = std::map<std::string, std::map<std::vector<std::tuple<int, int>>, real<T>>>();
         real<T> ws = spin();
         for (unsigned long long step = 0; step < total_step; step++) {
            ws = _markov_spin(ws);
            for (const auto& [kind, group] : observers) {
               for (const auto& [positions, tensor] : group) {
                  int body = positions.size();
                  auto current_spin = std::vector<int>();
                  for (auto i = 0; i < body; i++) {
                     current_spin.push_back(spin.configuration[std::get<0>(positions[i])][std::get<1>(positions[i])]);
                  }
                  real<T> value = 0;

                  for (const auto& [spins_out, element] : _find_element(*tensor).at(current_spin)) {
                     auto map = std::map<std::tuple<int, int>, int>();
                     for (auto i = 0; i < body; i++) {
                        map[positions[i]] = spins_out[i];
                     }
                     real<T> wss = spin(map);
                     value += element * wss / ws;
                  }
                  result[kind][positions] += value;
                  result_square[kind][positions] += value * value;
               }
            }
            if (calculate_energy) {
               real<T> energy = 0;
               real<T> energy_variance_square = 0;
               const auto& energy_square_pool = result_square.at("Energy");
               for (const auto& [positions, value] : result.at("Energy")) {
                  auto this_energy = value / (step + 1);
                  auto this_square = energy_square_pool.at(positions) / (step + 1);
                  energy += this_energy;
                  if (step != 0) {
                     energy_variance_square += (this_square - this_energy * this_energy) / (step + 1 - 1);
                  }
               };
               std::cout << clear_line << "Markov sampling, total_step=" << total_step << ", dimension=" << dimension_virtual
                         << ", dimension_cut=" << dimension_cut << ", step=" << step << ", Energy=" << energy / (M * N)
                         << " with sigma=" << std::sqrt(energy_variance_square) / (M * N) << "\r" << std::flush;
            } else {
               std::cout << clear_line << "Markov sampling, total_step=" << total_step << ", dimension=" << dimension_virtual
                         << ", dimension_cut=" << dimension_cut << ", step=" << step << "\r" << std::flush;
            }
         }
         for (auto& [kind, group] : result) {
            const auto& group_square = result_square.at(kind);
            for (auto& [positions, value] : group) {
               value /= total_step;
               auto value_square = group_square.at(positions);
               value_square /= total_step;
               result_variance_square[kind][positions] = (value_square - value * value) / (total_step - 1);
            }
         }
         if (calculate_energy) {
            real<T> energy = 0;
            real<T> energy_variance_square = 0;
            const auto& energy_variance_square_pool = result_variance_square.at("Energy");
            for (const auto& [positions, value] : result.at("Energy")) {
               energy += value;
               energy_variance_square += energy_variance_square_pool.at(positions);
            };
            std::cout << clear_line << "Markov sample done, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << ", Energy=" << energy / (M * N)
                      << " with sigma=" << std::sqrt(energy_variance_square) / (M * N) << "\n"
                      << std::flush;
         } else {
            std::cout << clear_line << "Markov sample done, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << "\n"
                      << std::flush;
         }
         return std::make_tuple(std::move(result), std::move(result_variance_square));
      }

      auto ergodic(
            std::map<std::string, std::map<std::vector<std::tuple<int, int>>, std::shared_ptr<const Tensor<T>>>> observers,
            bool calculate_energy = false) {
         std::cout << clear_line << "Ergodic sampling start, dimension=" << dimension_virtual << ", dimension_cut=" << dimension_cut << "\n"
                   << std::flush;
         if (calculate_energy) {
            observers["Energy"] = hamiltonians;
         }
         auto result = std::map<std::string, std::map<std::vector<std::tuple<int, int>>, real<T>>>();
         real<T> sum_of_ws_square = 0;
         unsigned long long total_step = std::pow(dimension_physics, M * N);
         for (unsigned long long step = 0; step < total_step; step++) {
            _ergodic_spin(step);
            real<T> ws = spin();
            sum_of_ws_square += ws * ws;
            for (const auto& [kind, group] : observers) {
               for (const auto& [positions, tensor] : group) {
                  int body = positions.size();
                  auto current_spin = std::vector<int>();
                  for (auto i = 0; i < body; i++) {
                     current_spin.push_back(spin.configuration[std::get<0>(positions[i])][std::get<1>(positions[i])]);
                  }
                  real<T> value = 0;

                  for (const auto& [spins_out, element] : _find_element(*tensor).at(current_spin)) {
                     auto map = std::map<std::tuple<int, int>, int>();
                     for (auto i = 0; i < body; i++) {
                        map[positions[i]] = spins_out[i];
                     }
                     real<T> wss = spin(map);
                     value += element * wss / ws;
                  }
                  result[kind][positions] += value * ws * ws;
               }
            }
            if (calculate_energy) {
               real<T> energy = 0;
               for (const auto& [positions, value] : result.at("Energy")) {
                  energy += value;
               };
               std::cout << clear_line << "Ergodic sampling, total_step=" << total_step << ", dimension=" << dimension_virtual
                         << ", dimension_cut=" << dimension_cut << ", step=" << step << ", Energy=" << energy / (sum_of_ws_square * M * N) << "\r"
                         << std::flush;
            } else {
               std::cout << clear_line << "Ergodic sampling, total_step=" << total_step << ", dimension=" << dimension_virtual
                         << ", dimension_cut=" << dimension_cut << ", step=" << step << "\r" << std::flush;
            }
         }
         if (calculate_energy) {
            real<T> energy = 0;
            for (const auto& [positions, value] : result.at("Energy")) {
               energy += value;
            };
            std::cout << clear_line << "Ergodic sample done, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << ", Energy=" << energy / (sum_of_ws_square * M * N) << "\n"
                      << std::flush;
         } else {
            std::cout << clear_line << "Ergodic sample done, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << "\n"
                      << std::flush;
         }
         for (auto& [kind, group] : result) {
            for (auto& [positions, value] : group) {
               value /= sum_of_ws_square;
            }
         }
         return result;
      }

      void _ergodic_spin(unsigned long long step) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               spin.set({i, j}, step % dimension_physics);
               step /= dimension_physics;
            }
         }
      }

      void equilibrate(unsigned long long total_step) {
         std::cout << clear_line << "Equilibrating start, total_step=" << total_step << ", dimension=" << dimension_virtual
                   << ", dimension_cut=" << dimension_cut << "\n"
                   << std::flush;
         real<T> ws = spin();
         for (unsigned long long step = 0; step < total_step; step++) {
            ws = _markov_spin(ws);
            std::cout << clear_line << "Equilibrating, total_step=" << total_step << ", dimension=" << dimension_virtual
                      << ", dimension_cut=" << dimension_cut << ", step=" << step << "\r" << std::flush;
         }
         std::cout << clear_line << "Equilibrate done, total_step=" << total_step << ", dimension=" << dimension_virtual
                   << ", dimension_cut=" << dimension_cut << "\n"
                   << std::flush;
      }

      real<T> _markov_spin(real<T> ws) {
         // TODO proper update order and use hint of aux
         for (auto iter = hamiltonians.begin(); iter != hamiltonians.end(); ++iter) {
            const auto& [positions, hamiltonian] = *iter;
            ws = _markov_single_term(ws, positions, hamiltonian);
         }
         for (auto iter = hamiltonians.rbegin(); iter != hamiltonians.rend(); ++iter) {
            const auto& [positions, hamiltonian] = *iter;
            ws = _markov_single_term(ws, positions, hamiltonian);
         }
         return ws;
      }

      real<T>
      _markov_single_term(real<T> ws, const std::vector<std::tuple<int, int>>& positions, const std::shared_ptr<const Tensor<T>>& hamiltonian) {
         int body = positions.size();
         auto current_spin = std::vector<int>();
         for (auto i = 0; i < body; i++) {
            current_spin.push_back(spin.configuration[std::get<0>(positions[i])][std::get<1>(positions[i])]);
         }
         const auto& hamiltonian_elements = _find_element(*hamiltonian);
         const auto& possible_hopping = hamiltonian_elements.at(current_spin);
         if (!possible_hopping.empty()) {
            int hopping_number = possible_hopping.size();
            auto random_index = random::uniform<int>(0, hopping_number - 1)();
            auto iter = possible_hopping.begin();
            for (auto i = 0; i < random_index; i++) {
               ++iter;
            }
            const auto& [spins_new, element] = *iter;
            auto replacement = std::map<std::tuple<int, int>, int>();
            for (auto i = 0; i < body; i++) {
               replacement[positions[i]] = spins_new[i];
            }
            real<T> wss = spin(replacement);
            int hopping_number_s = hamiltonian_elements.at(spins_new).size();
            real<T> wss_over_ws = wss / ws;
            real<T> p = wss_over_ws * wss_over_ws * hopping_number / hopping_number_s;
            if (random::uniform<real<T>>(0, 1)() < p) {
               ws = wss;
               for (auto i = 0; i < body; i++) {
                  spin.set(positions[i], spins_new[i]);
               }
            }
         }
         return ws;
      }

      inline static std::map<const Tensor<T>*, std::map<std::vector<int>, std::map<std::vector<int>, T>>> tensor_element_map = {};

      const std::map<std::vector<int>, std::map<std::vector<int>, T>>& _find_element(const Tensor<T>& tensor) const {
         auto tensor_id = &tensor;
         if (auto found = tensor_element_map.find(tensor_id); found != tensor_element_map.end()) {
            return found->second;
         }
         int body = tensor.names.size() / 2;
         auto& result = tensor_element_map[tensor_id];
         auto names = std::vector<Name>();
         auto index = std::vector<int>();
         for (auto i = 0; i < body; i++) {
            names.push_back("I" + std::to_string(i));
            index.push_back(0);
         }
         for (auto i = 0; i < body; i++) {
            names.push_back("O" + std::to_string(i));
            index.push_back(0);
         }
         while (true) {
            auto map = std::map<Name, Size>();
            for (auto i = 0; i < 2 * body; i++) {
               map[names[i]] = index[i];
            }
            auto value = tensor.const_at(map);
            if (value != 0) {
               auto spins_in = std::vector<int>();
               auto spins_out = std::vector<int>();
               for (auto i = 0; i < body; i++) {
                  spins_in.push_back(index[i]);
               }
               for (auto i = body; i < 2 * body; i++) {
                  spins_out.push_back(index[i]);
               }
               result[std::move(spins_in)][std::move(spins_out)] = value;
            }
            int active_position = 0;
            index[active_position] += 1;
            while (index[active_position] == dimension_physics) {
               index[active_position] = 0;
               active_position += 1;
               if (active_position == 2 * body) {
                  return result;
               }
               index[active_position] += 1;
            }
         }
      }
   };

   template<typename T>
   std::ostream& operator<(std::ostream& out, const SamplingGradientLattice<T>& lattice) {
      using TAT::operator<;
      out < static_cast<const AbstractNetworkLattice<T>&>(lattice);
      out < lattice.dimension_cut;
      out < lattice.spin.configuration;
      return out;
   }

   template<typename T>
   std::istream& operator>(std::istream& in, SamplingGradientLattice<T>& lattice) {
      using TAT::operator>;
      in > static_cast<AbstractNetworkLattice<T>&>(lattice);
      in > lattice.dimension_cut;
      std::vector<std::vector<int>> configuration;
      in > configuration;
      lattice.spin = SpinConfiguration(&lattice);
      lattice.initialize_spin(configuration);
      return in;
   }
} // namespace square

#endif
