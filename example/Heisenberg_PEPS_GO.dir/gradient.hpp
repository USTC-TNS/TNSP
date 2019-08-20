/* example/Heisenberg_PEPS_GO.dir/gradient.hpp
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

#ifndef TAT_GRADIENT_HPP_
#define TAT_GRADIENT_HPP_

#include <TAT.hpp>

#include "Configuration.hpp"
#include "PEPS.hpp"
#include "Markov.hpp"

struct Gradient {
    Markov_Engine& markov;
    Configuration& configuration;
    PEPS_GO& peps;

    Gradient(Markov_Engine& markov) : markov(markov), configuration(markov.configuration), peps(markov.peps) {}

    template<class Engine>
    void single_step(int markov_length, double step_size, Engine engine) {
        markov.reset();
        markov.chain(markov_length, engine);
        for (int i = 0; i < peps.metadata.M; i++) {
              for (int j = 0; j < peps.metadata.N; j++) {
                    for (TAT::Size k = 0; k < peps.metadata.d; k++) {
                          peps.lattice[{i, j, k}] -=
                                step_size * (markov.EsDelta[{i, j, k}] / markov_length -
                                             markov.Delta[{i, j, k}] * (markov.Es / (markov_length * markov_length)));
                    }
              }
        }
        std::cout << " E = " << markov.Es.at({}) / markov_length << "\n";
    }
};

#endif // TAT_GRADIENT_HPP_