/**
 * \file example/PEPS_GO.dir/Marov.hpp
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#ifndef TAT_MARKOV_HPP_
#define TAT_MARKOV_HPP_

#include <TAT.hpp>

#include "Configuration.hpp"
#include "PEPS.hpp"

struct Markov_Engine {
      Configuration& configuration;
      PEPS_GO& peps;
      Markov_Engine(Configuration& configuration) : configuration(configuration), peps(configuration.peps) {
            reset();
      }

      std::function<double(TAT::Size, TAT::Size, TAT::Size, TAT::Size)> hamiltonian;

      void reset() {
            auto set0 = []() { return 0; };
            Es = TAT::Node<double>(0);
            for (int i = 0; i < peps.metadata.M; i++) {
                  for (int j = 0; j < peps.metadata.M; j++) {
                        for (TAT::Size k = 0; k < peps.metadata.d; k++) {
                              Delta[{i, j, k}] =
                                    TAT::Node<double>(peps.legs_generator(i, j), peps.dims_generator(i, j)).set(set0);
                              EsDelta[{i, j, k}] =
                                    TAT::Node<double>(peps.legs_generator(i, j), peps.dims_generator(i, j)).set(set0);
                        }
                  }
            }
      }

      TAT::Node<double> Es;
      std::map<std::tuple<int, int, TAT::Size>, TAT::Node<double>> Delta;
      std::map<std::tuple<int, int, TAT::Size>, TAT::Node<double>> EsDelta;

      template<class Engine>
      void hop_next(Engine& engine) {
            auto total_hop_position = peps.metadata.M * (peps.metadata.N - 1) + (peps.metadata.M - 1) * peps.metadata.N;
            auto get_position = std::uniform_int_distribution<int>(0, total_hop_position - 1);
            auto position = get_position(engine);
            int x1, y1, x2, y2;
            TAT::Legs direction;
            if (position < peps.metadata.M * (peps.metadata.N - 1)) {
                  // 横的
                  y1 = position / peps.metadata.M;
                  x1 = position % peps.metadata.M;
                  direction = TAT::legs_name::Right;
                  x2 = x1;
                  y2 = y1 + 1;
            } else {
                  // 竖着
                  position -= peps.metadata.M * (peps.metadata.N - 1);
                  x1 = position / peps.metadata.N;
                  y1 = position % peps.metadata.N;
                  direction = TAT::legs_name::Down;
                  x2 = x1 + 1;
                  y2 = y1;
            }

            auto get_spin = std::uniform_int_distribution<TAT::Size>(0, peps.metadata.d * peps.metadata.d - 2);
            auto spin = get_spin(engine);
            auto current_spin = configuration.get_state(x1, y1) * peps.metadata.d + configuration.get_state(x2, y2);
            if (spin >= current_spin) {
                  spin += 1;
            }
            auto spin1 = spin / peps.metadata.d;
            auto spin2 = spin % peps.metadata.d;

            auto ws = configuration.ws().at({});
            auto wss = configuration.double_hole(x1, y1, direction, spin1, spin2).at({});
            auto possibility = (wss * wss) / (ws * ws);
            auto target_possibility = 1;
            if (possibility < target_possibility) {
                  auto get_random = std::uniform_real_distribution<double>(0, 1);
                  target_possibility = get_random(engine);
            }
            if (possibility > target_possibility) {
                  configuration.state[{x1, y1}].set_value(spin1);
                  configuration.state[{x2, y2}].set_value(spin2);
            }
      }

      enum struct SaveConfig { SaveNothing, SaveEnergy, SaveGradient };

      void save_current(SaveConfig config) {
            if (config == SaveConfig::SaveNothing) {
                  return;
            }
            auto E = TAT::Node<double>(0);
            for (int i = 0; i < peps.metadata.M; i++) {
                  for (int j = 0; j < peps.metadata.N - 1; j++) {
                        for (TAT::Size k = 0; k < peps.metadata.d; k++) {
                              for (TAT::Size l = 0; l < peps.metadata.d; l++) {
                                    auto Hss = hamiltonian(
                                          configuration.get_state(i, j), configuration.get_state(i, j + 1), k, l);
                                    if (Hss != 0) {
                                          E += Hss * configuration.double_hole(i, j, TAT::legs_name::Right, k, l) /
                                               configuration.ws();
                                    }
                              }
                        }
                  }
            }
            for (int i = 0; i < peps.metadata.M - 1; i++) {
                  for (int j = 0; j < peps.metadata.N; j++) {
                        for (TAT::Size k = 0; k < peps.metadata.d; k++) {
                              for (TAT::Size l = 0; l < peps.metadata.d; l++) {
                                    auto Hss = hamiltonian(
                                          configuration.get_state(i, j), configuration.get_state(i + 1, j), k, l);
                                    if (Hss != 0) {
                                          E += Hss * configuration.double_hole(i, j, TAT::legs_name::Down, k, l) /
                                               configuration.ws();
                                    }
                              }
                        }
                  }
            }
            Es += E;
            if (config == SaveConfig::SaveEnergy) {
                  return;
            }
            for (int i = 0; i < configuration.peps.metadata.M; i++) {
                  for (int j = 0; j < configuration.peps.metadata.N; j++) {
                        auto grad = configuration.gradient(i, j).transpose(peps.legs_generator(i, j));
                        Delta[{i, j, configuration.get_state(i, j)}] += grad;
                        EsDelta[{i, j, configuration.get_state(i, j)}] += grad * E;
                  }
            }
      }

      template<class Engine>
      void chain(int length, Engine& engine, SaveConfig config = SaveConfig::SaveGradient) {
            for (int i = 0; i < length; i++) {
                  hop_next(engine);
                  save_current(config);
            }
      }
};

#endif // TAT_MARKOV_HPP_
