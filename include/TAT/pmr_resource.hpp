/**
 * \file pmr_resource.hpp
 *
 * Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_PMR_RESOURCE_HPP
#define TAT_PMR_RESOURCE_HPP

#include <memory_resource>

namespace TAT {
   template<int buffer_size>
   struct scope_resource {
      std::byte buffer[buffer_size];
      std::pmr::monotonic_buffer_resource resource;
      std::pmr::memory_resource* upstream;
      scope_resource() : resource(buffer, sizeof(buffer)), upstream(std::pmr::set_default_resource(&resource)) {}
      ~scope_resource() {
         std::pmr::set_default_resource(upstream);
      }
   };

   template<typename buffer_type>
   struct scope_resource_adapter {
      buffer_type& buffer;
      std::pmr::monotonic_buffer_resource resource;
      std::pmr::memory_resource* upstream;
      scope_resource_adapter(buffer_type& buff) :
            buffer(buff), resource(buffer, sizeof(buffer)), upstream(std::pmr::set_default_resource(&resource)) {}
      ~scope_resource_adapter() {
         std::pmr::set_default_resource(upstream);
      }
   };
} // namespace TAT
#endif
