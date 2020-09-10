/**
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

#ifndef TAT_USE_MPI
#error testing mpi but mpi not enabled
#endif
#include <TAT/TAT.hpp>

using Tensor = TAT::Tensor<double, TAT::NoSymmetry>;

namespace mpi = TAT::mpi;

int main() {
   auto input = Tensor(mpi::mpi.rank);
   auto result = mpi::summary(input, mpi::mpi.size / 2);
   mpi::mpi_out(mpi::mpi.size / 2) << result << "\n";
   result = mpi::broadcast(result, mpi::mpi.size / 2);
   mpi::barrier();
   std::cout << mpi::mpi.rank << " " << result << "\n";
   // TODO cannot use endl
   return 0;
}
