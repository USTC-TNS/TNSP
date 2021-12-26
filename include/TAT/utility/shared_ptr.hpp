/**
 * \file shared_ptr.hpp
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
#ifndef TAT_SHARED_PTR_HPP
#define TAT_SHARED_PTR_HPP

#include <memory>

namespace TAT {
   namespace detail {
      template<typename T>
      struct shared_ptr_helper {
         T m_object;
         long m_count;
         template<typename... Args>
         shared_ptr_helper(Args&&... args) : m_object(std::forward<Args>(args)...), m_count(1) {}
      };

      /**
       * Alternative to std::propagate_const<std::shared_ptr<T>>
       *
       * Some useless feature removed
       */
      template<typename T>
      struct shared_ptr {
         shared_ptr_helper<T>* m_pointer;

         // see https://zh.cppreference.com/w/cpp/memory/shared_ptr

         // (1)
         // modified by myself
         shared_ptr(shared_ptr_helper<T>* pointer = nullptr) : m_pointer(pointer) {}
         // (2)
         // deleted by myself
         // shared_ptr(std::nullptr_t) : m_pointer(nullptr) {}
         // (3-7) without polymorphism, deleter and allocator
         // deleted by myself
         // shared_ptr(T* ptr) : m_pointer(ptr ? new shared_ptr_helper<T>{ptr, 1} : nullptr) {}
         // (8) no such constructor
         // (9) without polymorphism
         shared_ptr(const shared_ptr<T>& other) : m_pointer(other.m_pointer) {
            if (m_pointer) {
               m_pointer->m_count++;
            }
         }
         // (10) without polymorphism
         shared_ptr(shared_ptr<T>&& other) : m_pointer(other.m_pointer) {
            other.m_pointer = nullptr;
         }
         // (11-13) no such constructor

         ~shared_ptr() {
            if (m_pointer) {
               m_pointer->m_count--;
               if (m_pointer->m_count == 0) {
                  // delete m_pointer->m_object;
                  delete m_pointer;
               }
            }
         }

         // (1)
         shared_ptr<T>& operator=(const shared_ptr<T>& other) {
            this->~shared_ptr();
            new (this) shared_ptr<T>(other);
            return *this;
         }
         // (2)
         shared_ptr<T>& operator=(shared_ptr<T>&& other) {
            if (this != &other) {
               this->~shared_ptr();
               new (this) shared_ptr<T>(std::move(other));
            }
            return *this;
         }
         // (3-4) no such operator=

         // reset, swap I do not use it

         const T* get() const noexcept {
            return &(m_pointer->m_object);
         }
         T* get() noexcept {
            return &(m_pointer->m_object);
         }
         const T& operator*() const noexcept {
            return *get();
         }
         T& operator*() noexcept {
            return *get();
         }
         const T* operator->() const noexcept {
            return get();
         }
         T* operator->() noexcept {
            return get();
         }

         long use_count() const {
            if (m_pointer) {
               return m_pointer->m_count;
            } else {
               return 0;
            }
         }

         template<typename... Args>
         static shared_ptr<T> make(Args&&... args) {
            // T* object = new T(std::forward<Args>(args)...);
            // return shared_ptr<T>(object);
            auto pointer = new shared_ptr_helper<T>(std::forward<Args>(args)...);
            return shared_ptr<T>(pointer);
         }
      };
   } // namespace detail
} // namespace TAT
#endif
