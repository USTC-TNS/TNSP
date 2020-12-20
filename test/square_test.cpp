/**
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

// #define TAT_USE_SIMPLE_NAME
// #define LAZY_DEBUG

#include <fstream>
#include <square/square.hpp>

enum struct LatticeType { Exact, Simple, Sample };

int main(int argc, char** argv) {
   std::ifstream config_file;
   if (argc != 1) {
      config_file.open(argv[1]);
      if (config_file.is_open()) {
         std::cin.rdbuf(config_file.rdbuf());
      }
   } else {
      config_file.open("INPUT");
      if (config_file.is_open()) {
         std::cin.rdbuf(config_file.rdbuf());
      }
   }
   // lattice
   LatticeType lattice_type = LatticeType::Simple;
   square::ExactLattice<double> exact_lattice;
   square::SimpleUpdateLattice<double> simple_update_lattice;
   square::SamplingGradientLattice<double> sampling_gradient_lattice;
   while (true) {
      std::string command;
      std::cin >> command;
      if (command == "") {
         return 0;
      }
      if (command == "use_exact") {
         lattice_type = LatticeType::Exact;
         continue;
      } else if (command == "use_simple") {
         lattice_type = LatticeType::Simple;
         continue;
      } else if (command == "use_sample") {
         lattice_type = LatticeType::Sample;
         continue;
      } else if (command == "seed") {
         unsigned long seed;
         std::cin >> seed;
         square::random::seed(seed);
         continue;
      }
      switch (lattice_type) {
         case LatticeType::Exact:
            if (command == "save") {
               std::string file_name;
               std::cin >> file_name;
               std::ofstream(file_name) < exact_lattice;
            } else if (command == "open") {
               std::string file_name;
               std::cin >> file_name;
               std::ifstream(file_name) > exact_lattice;
            } else if (command == "new") {
               int M, N;
               square::Size d;
               std::cin >> M >> N >> d;
               exact_lattice = square::ExactLattice<double>(M, N, d);
               exact_lattice.set_all_horizontal_bond(square::Common<double>::SS());
               exact_lattice.set_all_vertical_bond(square::Common<double>::SS());
            } else if (command == "update") {
               int total_step;
               square::real<double> approximate_energy;
               std::cin >> total_step >> approximate_energy;
               exact_lattice.update(total_step, approximate_energy);
            } else if (command == "energy") {
               std::cout << exact_lattice.observe_energy() << "\n";
            } else {
               std::cerr << "Invalid Command: " << command << "\n";
               return -1;
            }
            break;
         case LatticeType::Simple:
            if (command == "save") {
               std::string file_name;
               std::cin >> file_name;
               std::ofstream(file_name) < simple_update_lattice;
            } else if (command == "open") {
               std::string file_name;
               std::cin >> file_name;
               std::ifstream(file_name) > simple_update_lattice;
            } else if (command == "new") {
               int M, N;
               square::Size D, d;
               std::cin >> M >> N >> D >> d;
               simple_update_lattice = square::SimpleUpdateLattice<double>(M, N, D, d);
               simple_update_lattice.set_all_horizontal_bond(square::Common<double>::SS());
               simple_update_lattice.set_all_vertical_bond(square::Common<double>::SS());
            } else if (command == "update") {
               int total_step;
               square::real<double> delta_t;
               square::Size new_dimension;
               std::cin >> total_step >> delta_t >> new_dimension;
               simple_update_lattice.update(total_step, delta_t, new_dimension);
            } else if (command == "exact") {
               lattice_type = LatticeType::Exact;
               exact_lattice = square::ExactLattice(simple_update_lattice);
            } else if (command == "sample") {
               square::Size Dc;
               std::cin >> Dc;
               lattice_type = LatticeType::Sample;
               sampling_gradient_lattice = square::SamplingGradientLattice<double>(simple_update_lattice, Dc);
            } else {
               std::cerr << "Invalid Command: " << command << "\n";
               return -1;
            }
            break;
         case LatticeType::Sample:
            if (command == "save") {
               std::string file_name;
               std::cin >> file_name;
               std::ofstream(file_name) < sampling_gradient_lattice;
            } else if (command == "open") {
               std::string file_name;
               std::cin >> file_name;
               std::ifstream(file_name) > sampling_gradient_lattice;
            } else if (command == "new") {
               int M, N;
               square::Size D, Dc, d;
               std::cin >> M >> N >> D >> Dc >> d;
               sampling_gradient_lattice = square::SamplingGradientLattice<double>(M, N, D, Dc, d);
               sampling_gradient_lattice.set_all_horizontal_bond(square::Common<double>::SS());
               sampling_gradient_lattice.set_all_vertical_bond(square::Common<double>::SS());
            } else if (command == "ergodic") {
               sampling_gradient_lattice.ergodic({}, true);
            } else if (command == "markov") {
               unsigned long long total_step;
               std::cin >> total_step;
               sampling_gradient_lattice.markov(total_step, {}, true);
            } else if (command == "gradient") {
               unsigned long long gradient_step, markov_step;
               square::real<double> step_size;
               std::cin >> gradient_step >> step_size >> markov_step;
               std::cout << "Gradient descent start, total_step=" << gradient_step << "\n" << std::flush;
               std::cout << "\n\n\n";
               for (unsigned long long step = 0; step < gradient_step; step++) {
                  std::cout << "\u001b[3A"
                            << "Gradient descenting, total_step=" << gradient_step << ", step=" << step << "\n"
                            << std::flush;
                  auto [energy, variance, gradient] = sampling_gradient_lattice.markov(markov_step, {}, true, true);
                  for (auto i = 0; i < sampling_gradient_lattice.M; i++) {
                     for (auto j = 0; j < sampling_gradient_lattice.N; j++) {
                        sampling_gradient_lattice.lattice[i][j] -= step_size * gradient[i][j];
                        sampling_gradient_lattice.lattice[i][j] /= sampling_gradient_lattice.lattice[i][j].norm<-1>();
                     }
                  }
               }
               std::cout << square::clear_line << "Gradient descent done, total_step=" << gradient_step << "\n" << std::flush;
            } else if (command == "equilibrate") {
               sampling_gradient_lattice.initialize_spin([](int i, int j) { return (i + j) % 2; });
               unsigned long long total_step;
               std::cin >> total_step;
               sampling_gradient_lattice.equilibrate(total_step);
            } else if (command == "exact") {
               lattice_type = LatticeType::Exact;
               exact_lattice = square::ExactLattice(sampling_gradient_lattice);
            } else if (command == "simple") {
               lattice_type = LatticeType::Simple;
               simple_update_lattice = square::SimpleUpdateLattice<double>(sampling_gradient_lattice);
            } else {
               std::cerr << "Invalid Command: " << command << "\n";
               return -1;
            }
            break;
      }
   }
   return 0;
}
