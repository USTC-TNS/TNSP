/**
 * \file mpi.hpp
 *
 * Copyright (C) 2019-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_MPI_HPP
#define TAT_MPI_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../utility/timer.hpp"
#include "io.hpp"

#ifdef TAT_USE_MPI
// Can not use extern "C" here, since some version of mpi will detect macro __cplusplus
// and then it will declare some c++ function, and this macro cannot be mask
#include <mpi.h>
#endif

#ifdef _WIN32
// windows.h will define macro min and max by default which is annoying
#define NOMINMAX
#include <windows.h>
#endif

namespace TAT {
    /**
     * Wrapper for ostream, the stream will only output in the rank which is specified when creating
     */
    struct mpi_one_output_stream {
        std::ostream& out;
        bool valid;
        std::ostringstream string;
        ~mpi_one_output_stream() {
            out << string.str() << std::flush;
        }
        mpi_one_output_stream(std::ostream& out, bool valid) : out(out), valid(valid) { }

        template<typename Type>
        mpi_one_output_stream& operator<<(const Type& value) & {
            if (valid) {
                string << value;
            }
            return *this;
        }

        template<typename Type>
        mpi_one_output_stream&& operator<<(const Type& value) && {
            if (valid) {
                string << value;
            }
            return std::move(*this);
        }
    };

    /**
     * Wrapper for ostream, the stream will output the rank which is psecified when creating
     */
    struct mpi_rank_output_stream {
        std::ostream& out;
        std::ostringstream string;
        ~mpi_rank_output_stream() {
            out << string.str() << std::flush;
        }
        mpi_rank_output_stream(std::ostream& out, int rank) : out(out) {
            if (rank != -1) {
                string << "[rank " << rank << "] ";
            }
        }

        template<typename Type>
        mpi_rank_output_stream& operator<<(const Type& value) & {
            string << value;
            return *this;
        }

        template<typename Type>
        mpi_rank_output_stream&& operator<<(const Type& value) && {
            string << value;
            return std::move(*this);
        }
    };

    namespace detail {
        template<typename T>
        using serializable_helper = std::
            pair<decltype(std::declval<std::ostream&>() < std::declval<const T&>()), decltype(std::declval<std::istream&>() > std::declval<T&>())>;
    } // namespace detail
    template<typename T>
    constexpr bool serializable = is_detected_v<detail::serializable_helper, T>;

    inline timer mpi_send_guard("mpi_send");
    inline timer mpi_receive_guard("mpi_receive");
    inline timer mpi_broadcast_guard("mpi_broadcast");
    inline timer mpi_reduce_guard("mpi_reduce");

    /**
     * MPI handler type
     *
     * It will call MPI_Init and MPI_Finalize automatically, and get size and rank information.
     * It also supply ostream related to mpi rank
     *
     * \note creating multiple MPI handler will not crash the program
     */
    struct mpi_t {
        int size = 1;
        int rank = 0;
#ifdef TAT_USE_MPI
        static constexpr bool enabled = true;
#else
        static constexpr bool enabled = false;
#endif
#ifdef TAT_USE_MPI
        static bool initialized() {
            int result;
            MPI_Initialized(&result);
            return result;
        }
        static bool finalized() {
            int result;
            MPI_Finalized(&result);
            return result;
        }
        mpi_t() {
            if (!initialized()) {
                MPI_Init(nullptr, nullptr);
            }
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
        ~mpi_t() {
            if (!finalized()) {
                MPI_Finalize();
            }
        }

        static void barrier() {
            MPI_Barrier(MPI_COMM_WORLD);
        }

        static constexpr int mpi_tag = 0;

        template<typename Type, typename = std::enable_if_t<serializable<Type>>>
        static void send(const Type& value, const int destination) {
            auto timer_guard = mpi_send_guard();
            detail::basic_outstringstream<char> stream;
            stream < value;
            auto data = std::move(stream).str();
            MPI_Send(data.data(), data.length(), MPI_BYTE, destination, mpi_tag, MPI_COMM_WORLD);
        }

        template<typename Type, typename = std::enable_if_t<serializable<Type>>>
        static Type receive(const int source) {
            auto timer_guard = mpi_receive_guard();
            auto status = MPI_Status();
            MPI_Probe(source, mpi_tag, MPI_COMM_WORLD, &status);
            int length;
            MPI_Get_count(&status, MPI_BYTE, &length);
            auto data = std::string();
            data.resize(length);
            MPI_Recv(data.data(), length, MPI_BYTE, source, mpi_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            detail::basic_instringstream<char> stream(data);
            auto result = Type();
            stream > result;
            return result;
        }

        template<typename Type, typename = std::enable_if_t<serializable<Type>>>
        Type send_receive(const Type& value, const int source, const int destination) const {
            if (rank == source) {
                send(value, destination);
            }
            if (rank == destination) {
                return receive<Type>(source);
            }
            return Type();
        }

        template<typename Type, typename = std::enable_if_t<serializable<Type>>>
        Type broadcast(const Type& value, const int root) const {
            auto timer_guard = mpi_broadcast_guard();
            if (size == 1) {
                return value;
            }
            // there is many time wasted in mpi communication so there is no constexpr if here again
            if (0 > root || root >= size) {
                detail::error("Invalid root rank when mpi broadcast");
            }
            const auto this_fake_rank = (size + rank - root) % size;
            Type result;
            // get from father
            if (this_fake_rank != 0) {
                const auto father_fake_rank = (this_fake_rank - 1) / 2;
                const auto father_real_rank = (father_fake_rank + root) % size;
                result = receive<Type>(father_real_rank);
            } else {
                // if itself is root, copy the value
                result = value;
            }
            // send to son
            const auto left_son_fake_rank = this_fake_rank * 2 + 1;
            const auto right_son_fake_rank = this_fake_rank * 2 + 2;
            if (left_son_fake_rank < size) {
                const auto left_son_real_rank = (left_son_fake_rank + root) % size;
                send(result, left_son_real_rank);
            }
            if (right_son_fake_rank < size) {
                const auto right_son_real_rank = (right_son_fake_rank + root) % size;
                send(result, right_son_real_rank);
            }
            return result;
        }

        template<typename Type, typename Func, typename = std::enable_if_t<serializable<Type> && std::is_invocable_r_v<Type, Func, Type, Type>>>
        Type reduce(const Type& value, const int root, Func&& function) const {
            auto timer_guard = mpi_reduce_guard();
            if (size == 1) {
                return value;
            }
            if (0 > root || root >= size) {
                detail::error("Invalid root rank when mpi reduce");
            }
            const auto this_fake_rank = (size + rank - root) % size;
            Type result;
            // get from son
            const auto left_son_fake_rank = this_fake_rank * 2 + 1;
            const auto right_son_fake_rank = this_fake_rank * 2 + 2;
            if (left_son_fake_rank < size) {
                const auto left_son_real_rank = (left_son_fake_rank + root) % size;
                result = function(value, receive<Type>(left_son_real_rank));
            }
            if (right_son_fake_rank < size) {
                const auto right_son_real_rank = (right_son_fake_rank + root) % size;
                // if left son does not exist, then right son does not exist definitely
                // so it does not need to check validity of result
                result = function(result, receive<Type>(right_son_real_rank));
            }
            // pass to father
            if (this_fake_rank != 0) {
                const auto father_fake_rank = (this_fake_rank - 1) / 2;
                const auto father_real_rank = (father_fake_rank + root) % size;
                if (left_son_fake_rank < size) {
                    send(result, father_real_rank);
                } else {
                    send(value, father_real_rank);
                }
            }
            return result;
        }
#endif
        auto out_one(int rank_specified = 0) {
            return mpi_one_output_stream(std::cout, rank_specified == rank);
        }
        auto log_one(int rank_specified = 0) {
            return mpi_one_output_stream(std::clog, rank_specified == rank);
        }
        auto err_one(int rank_specified = 0) {
            return mpi_one_output_stream(std::cerr, rank_specified == rank);
        }

        auto out_rank() {
            return mpi_rank_output_stream(std::cout, size == 1 ? -1 : rank);
        }
        auto log_rank() {
            return mpi_rank_output_stream(std::clog, size == 1 ? -1 : rank);
        }
        auto err_rank() {
            return mpi_rank_output_stream(std::cerr, size == 1 ? -1 : rank);
        }
    };
    inline mpi_t mpi;

    inline detail::evil_t::evil_t() {
#ifdef _WIN32
        HANDLE output_handle = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD output_mode = 0;
        GetConsoleMode(output_handle, &output_mode);
        SetConsoleMode(output_handle, output_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        HANDLE error_handle = GetStdHandle(STD_ERROR_HANDLE);
        DWORD error_mode = 0;
        GetConsoleMode(error_handle, &error_mode);
        SetConsoleMode(error_handle, error_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
    }
    inline detail::evil_t::~evil_t() { }

    inline void detail::log(const char* message) {
        mpi.log_rank() << console_yellow << message << console_origin << '\n';
    }

    inline void detail::warning(const char* message) {
        mpi.err_rank() << console_red << message << console_origin << '\n';
    }

    inline void detail::error(const char* message) {
        throw std::runtime_error(message);
    }
} // namespace TAT
#endif
