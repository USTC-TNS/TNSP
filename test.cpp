/* TAT
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

#define TAT_USE_CPU

// SVD
// #define TAT_USE_GESDD
// #define TAT_USE_GESVD
#define TAT_USE_GESVDX

// QR
#define TAT_USE_GEQRF
// #define TAT_USE_GEQP3
// GEQP3 not understand, maybe useful if R will drop

#define TAT_TEST_MAIN

#include "TAT.hpp"
