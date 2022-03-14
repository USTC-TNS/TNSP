/**
 * \file TAT.hpp
 *
 * Copyright (C) 2019-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_HPP
#define TAT_HPP

#ifndef __cplusplus
#error only work for c++
#endif

#if __cplusplus < 201703L
#error require c++17 or later
#endif

#include "structure/tensor.hpp"

#include "miscellaneous/io.hpp"
#include "miscellaneous/mpi.hpp"
#include "miscellaneous/scalar.hpp"

#include "implement/contract.hpp"
#include "implement/edge_miscellaneous.hpp"
#include "implement/edge_operator.hpp"
#include "implement/exponential.hpp"
#include "implement/get_item_and_clear_symmetry.hpp"
#include "implement/identity_and_conjugate.hpp"
#include "implement/qr.hpp"
#include "implement/shrink_and_expand.hpp"
#include "implement/svd.hpp"
#include "implement/trace.hpp"
#include "implement/transpose.hpp"

#endif
