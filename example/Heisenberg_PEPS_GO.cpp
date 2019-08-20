/* example/Heisenberg_PEPS_GO.cpp
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

#include <random>

#include <args.hxx>

#define TAT_DEFAULT
#include <TAT.hpp>

#include "Heisenberg_PEPS_GO.dir/Configuration.hpp"
#include "Heisenberg_PEPS_GO.dir/Markov.hpp"
#include "Heisenberg_PEPS_GO.dir/PEPS.hpp"
#include "Heisenberg_PEPS_GO.dir/gradient.hpp"

int main() {
      std::ios_base::sync_with_stdio(false);

      auto engine = std::default_random_engine(0);
      auto dist = std::uniform_real_distribution<double>(-1, 1);
      auto generator = [&dist, &engine]() { return dist(engine); };

      auto peps = PEPS_GO();
      peps.initial_metadata(4, 4, 2, 4); // M=4, N=4, d=2, D=4
      peps.initial_lattice();
      peps.set_random_lattice(generator);

      auto configuration = Configuration(peps);
      configuration.initial_aux(8, 2, generator); // Dc=8, scan_time=2
      configuration.calculate_hole();

      auto state = std::map<std::tuple<int, int>, TAT::Size>();
      for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                  state[{i, j}] = (i + j) % 2;
            }
      }
      configuration.set_state(state);

      auto markov = Markov_Engine(configuration);
      markov.hamiltonian = [](TAT::Size a, TAT::Size b, TAT::Size c, TAT::Size d) {
            if (a == c && b == d) {
                  if (a != b) {
                        return -1 / 4.;
                  } else {
                        return +1 / 4.;
                  }
            }
            if (a == d && b == c) {
                  if (a != b) {
                        return +2 / 4.;
                  }
            }
            return 0.;
      };

      auto grad = Gradient(markov);
      for (int step = 0; step < 1000; step++) {
            std::cout << " step = " << step;
            grad.single_step(10, 0.1, engine);
      }
}
