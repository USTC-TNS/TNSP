/**
 * \file io.hpp
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
#ifndef TAT_IO_HPP
#define TAT_IO_HPP

#include <iostream>
#include <limits>
#include <type_traits>

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   namespace detail {
      template<typename char_type>
      class basic_instringstream :
            private std::basic_streambuf<char_type, std::char_traits<char_type>>,
            public std::basic_istream<char_type, std::char_traits<char_type>> {
         using traits_type = std::char_traits<char_type>;
         using base_buf_type = std::basic_streambuf<char_type, traits_type>;
         using base_stream_type = std::basic_istream<char_type, traits_type>;
         using int_type = typename base_buf_type::int_type;

         std::basic_string_view<char_type> m_str;

       public:
         explicit basic_instringstream(std::basic_string_view<char_type> str) : base_stream_type(this), m_str(str) {
            this->setg(const_cast<char_type*>(&*m_str.begin()), const_cast<char_type*>(&*m_str.begin()), const_cast<char_type*>(&*m_str.end()));
         }
      };

      template<typename char_type>
      class basic_outstringstream :
            private std::basic_streambuf<char_type, std::char_traits<char_type>>,
            public std::basic_ostream<char_type, std::char_traits<char_type>> {
         using traits_type = std::char_traits<char_type>;
         using base_buf_type = std::basic_streambuf<char_type, traits_type>;
         using base_stream_type = std::basic_ostream<char_type, traits_type>;
         using int_type = typename base_buf_type::int_type;

         std::basic_string<char_type> m_str;

         int_type overflow(int_type ch) override {
            // pbase, pptr, epptr
            if (traits_type::eq_int_type(ch, traits_type::eof())) {
               return ch;
            }

            m_str.resize(m_str.size() * 2);

            const std::ptrdiff_t diff = this->pptr() - this->pbase();
            this->setp(&*m_str.begin(), &*m_str.end());
            this->pbump(diff);

            *this->pptr() = traits_type::to_char_type(ch);
            this->pbump(1);

            return ch; // return any value except eof
         }

       public:
         explicit basic_outstringstream(std::size_t size = 8) : base_stream_type(this) {
            m_str.resize(size);
            this->setp(&*m_str.begin(), &*m_str.end());
         }

         std::basic_string<char_type> str() && {
            const std::ptrdiff_t diff = this->pptr() - this->pbase();
            m_str.resize(diff);
            return std::move(m_str);
         }
      };

      // complex text io, complex bin io can be done directly
      template<typename ScalarType>
      std::ostream& print_complex(std::ostream& out, const std::complex<ScalarType>& value) {
         if (value.real() != 0) {
            out << value.real();
            if (value.imag() != 0) {
               if (value.imag() > 0) {
                  out << '+';
               }
               out << value.imag();
               out << 'i';
            }
         } else {
            if (value.imag() == 0) {
               out << '0';
            } else {
               out << value.imag();
               out << 'i';
            }
         }
         return out;
      }

      template<typename ScalarType>
      std::ostream&& print_complex(std::ostream&& out, const std::complex<ScalarType>& value) {
         print_complex(out, value);
         return std::move(out);
      }

      template<typename ScalarType>
      std::istream& scan_complex(std::istream& in, std::complex<ScalarType>& value) {
         ScalarType part;
         in >> part;
         char maybe_i = in.peek();
         if (maybe_i == 'i') {
            in.get();
            // no real part
            value = std::complex<ScalarType>{0, part};
         } else {
            // have real part
            if (maybe_i == '+' || maybe_i == '-') {
               // have imag part
               ScalarType another_part;
               in >> another_part;
               value = std::complex<ScalarType>{part, another_part};
               if (in.get() != 'i') {
                  in.setstate(std::ios::failbit);
               }
            } else {
               // no imag part
               value = std::complex<ScalarType>{part, 0};
            }
         }
         return in;
      }

      template<typename ScalarType>
      std::istream&& scan_complex(std::istream&& in, std::complex<ScalarType>& value) {
         detail::scan_complex(in, value);
         return std::move(in);
      }

      inline void ignore_until(std::istream& in, char end) {
         in.ignore(std::numeric_limits<std::streamsize>::max(), end);
      }

      template<typename Func>
      std::ostream& print_list(std::ostream& out, Func&& print, char left, char right) {
         out << left;
         while (!print(out)) {
            out << ',';
         }
         out << right;
         return out;
      }

      template<typename Func>
      std::istream& scan_list(std::istream& in, Func&& scan, char left, char right) {
         detail::ignore_until(in, left);
         if (in.peek() == right) {
            // empty list
            in.get(); // get ']'
         } else {
            // not empty
            do {
               scan(in);
            } while (in.get() == ',');
         }
         return in;
      }
   } // namespace detail

   // trivial type bin io

   template<typename T, typename = std::enable_if_t<std::is_trivially_destructible_v<T>>>
   std::ostream& operator<(std::ostream& out, const T& data) {
      out.write(reinterpret_cast<const char*>(&data), sizeof(T));
      return out;
   }
   template<typename T, typename = std::enable_if_t<std::is_trivially_destructible_v<T>>>
   std::istream& operator>(std::istream& in, T& data) {
      in.read(reinterpret_cast<char*>(&data), sizeof(T));
      return in;
   }

   // name type io

   namespace detail {
      inline bool valid_name_character(char c) {
         if (!std::isprint(c)) {
            return false;
         }
         if (c == ' ') {
            return false;
         }
         if (c == ',') {
            return false;
         }
         if (c == '[') {
            return false;
         }
         if (c == ']') {
            return false;
         }
         return true;
      }
   } // namespace detail

   inline std::ostream& print_string_for_name(std::ostream& out, const std::string& name) {
      return out << name;
   }
   inline std::ostream& print_fastname_for_name(std::ostream& out, const FastName& name) {
      out << static_cast<const std::string&>(name);
      return out;
   }

   inline std::istream& scan_string_for_name(std::istream& in, std::string& name) {
      char buffer[256]; // max name length = 256
      Size length = 0;
      while (detail::valid_name_character(in.peek())) {
         buffer[length++] = in.get();
      }
      buffer[length] = '\x00';
      name = (const char*)buffer;
      return in;
   }

   inline std::istream& scan_fastname_for_name(std::istream& in, FastName& name) {
      std::string string;
      scan_string_for_name(in, string);
      name = FastName(string);
      return in;
   }

   inline std::ostream& write_string_for_name(std::ostream& out, const std::string& string) {
      Size count = string.size();
      out < count;
      out.write(string.data(), sizeof(char) * count);
      return out;
   }
   inline std::istream& read_string_for_name(std::istream& in, std::string& string) {
      Size count;
      in > count;
      string.resize(count);
      in.read(string.data(), sizeof(char) * count);
      return in;
   }

   inline std::ostream& write_fastname_for_name(std::ostream& out, const FastName& name) {
      return write_string_for_name(out, static_cast<const std::string&>(name));
   }

   inline std::istream& read_fastname_for_name(std::istream& in, FastName& name) {
      std::string name_string;
      read_string_for_name(in, name_string);
      name = FastName(name_string);
      return in;
   }

   template<>
   struct NameTraits<FastName> {
      // Although FastName is trivial type, but write string explicitly for good compatibility.
      static constexpr out_operator_t<FastName> write = write_fastname_for_name;
      static constexpr in_operator_t<FastName> read = read_fastname_for_name;
      static constexpr out_operator_t<FastName> print = print_fastname_for_name;
      static constexpr in_operator_t<FastName> scan = scan_fastname_for_name;
   };
   template<>
   struct NameTraits<std::string> {
      static constexpr out_operator_t<std::string> write = write_string_for_name;
      static constexpr in_operator_t<std::string> read = read_string_for_name;
      static constexpr out_operator_t<std::string> print = print_string_for_name;
      static constexpr in_operator_t<std::string> scan = scan_string_for_name;
   };

   // vector io, bin and text

   template<typename T, typename A>
   std::ostream& operator<(std::ostream& out, const std::vector<T, A>& list) {
      Size count = list.size();
      out < count;
      if constexpr (is_name<T>) {
         for (const auto& i : list) {
            NameTraits<T>::write(out, i);
         }
      } else if constexpr (std::is_trivially_destructible_v<T>) {
         out.write(reinterpret_cast<const char*>(list.data()), sizeof(T) * count);
      } else {
         for (const auto& i : list) {
            out < i;
         }
      }
      return out;
   }
   template<typename T, typename A>
   std::istream& operator>(std::istream& in, std::vector<T, A>& list) {
      list.clear();
      Size count;
      in > count;
      if constexpr (is_name<T>) {
         for (Size i = 0; i < count; i++) {
            auto& item = list.emplace_back();
            NameTraits<T>::read(in, item);
         }
      } else if constexpr (std::is_trivially_destructible_v<T>) {
         list.resize(count);
         in.read(reinterpret_cast<char*>(list.data()), sizeof(T) * count);
      } else {
         for (Size i = 0; i < count; i++) {
            auto& item = list.emplace_back();
            in > item;
         }
      }
      return in;
   }

   template<typename T, typename A, typename = std::enable_if_t<is_scalar<T> || is_edge<T> || is_symmetry<T> || is_name<T>>>
   std::ostream& operator<<(std::ostream& out, const std::vector<T, A>& list) {
      detail::print_list(
            out,
            [offset = 0, l = list.data(), count = list.size()](std::ostream& out) mutable {
               if (offset == count) {
                  return true;
               }
               if constexpr (is_name<T>) {
                  NameTraits<T>::print(out, l[offset]);
               } else if constexpr (is_complex<T>) {
                  detail::print_complex(out, l[offset]);
               } else {
                  out << l[offset];
               }
               ++offset;
               return offset == count;
            },
            '[',
            ']');
      return out;
   }

   template<typename T, typename A, typename = std::enable_if_t<is_scalar<T> || is_edge<T> || is_symmetry<T> || is_name<T>>>
   std::istream& operator>>(std::istream& in, std::vector<T, A>& list) {
      list.clear();
      detail::scan_list(
            in,
            [&l = list](std::istream& in) mutable {
               T& i = l.emplace_back();
               if constexpr (is_name<T>) {
                  NameTraits<T>::scan(in, i);
               } else if constexpr (is_complex<T>) {
                  detail::scan_complex(in, i);
               } else {
                  in >> i;
               }
            },
            '[',
            ']');
      return in;
   }

   // edge io

   template<typename Symmetry>
   std::ostream& operator<<(std::ostream& out, const Edge<Symmetry>& edge) {
      if constexpr (Symmetry::length == 0) {
         out << edge.segments().front().second;
      } else {
         if constexpr (Symmetry::is_fermi_symmetry) {
            out << '{';
            out << "arrow" << ':';
            out << edge.arrow();
            out << ',';
            out << "segment" << ':';
         }
         detail::print_list(
               out,
               [offset = 0, l = edge.segments().data(), count = edge.segments().size()](std::ostream& out) mutable {
                  if (offset == count) {
                     return true;
                  }
                  const auto& [symmetry, dimension] = l[offset];
                  out << symmetry << ':' << dimension;
                  ++offset;
                  return offset == count;
               },
               '{',
               '}');
         if constexpr (Symmetry::is_fermi_symmetry) {
            out << '}';
         }
      }
      return out;
   }
   template<typename Symmetry>
   std::istream& operator>>(std::istream& in, Edge<Symmetry>& edge) {
      if constexpr (Symmetry::length == 0) {
         Size dimension;
         in >> dimension;
         edge = Edge<Symmetry>(dimension);
      } else {
         bool arrow = false;
         if constexpr (Symmetry::is_fermi_symmetry) {
            detail::ignore_until(in, ':');
            in >> arrow;
         }
         std::vector<std::pair<Symmetry, Size>> segments;
         detail::scan_list(
               in,
               [&l = segments](std::istream& in) mutable {
                  Symmetry symmetry;
                  in >> symmetry;
                  detail::ignore_until(in, ':');
                  Size dimension;
                  in >> dimension;
                  l.emplace_back(symmetry, dimension);
               },
               '{',
               '}');
         edge = {std::move(segments), arrow};
         if constexpr (Symmetry::is_fermi_symmetry) {
            detail::ignore_until(in, '}');
         }
      }
      return in;
   }

   template<typename Symmetry>
   std::ostream& operator<(std::ostream& out, const Edge<Symmetry>& edge) {
      if constexpr (Symmetry::is_fermi_symmetry) {
         out < edge.arrow();
      }
      out < edge.segments();
      return out;
   }
   template<typename Symmetry>
   std::istream& operator>(std::istream& in, Edge<Symmetry>& edge) {
      bool arrow = false;
      if constexpr (Symmetry::is_fermi_symmetry) {
         in > arrow;
      }
      std::vector<std::pair<Symmetry, Size>> segments;
      in > segments;
      edge = {std::move(segments), arrow};
      return in;
   }

   // symmetry io, text only, bin can use direct

   namespace detail {
      template<typename Symmetry, std::size_t... Is>
      void print_symmetry_sequence(std::ostream& out, const Symmetry& symmetry, std::index_sequence<Is...>) {
         (((Is == 0 ? out : out << ',') << std::get<Is>(symmetry)), ...);
      }

      template<typename Symmetry, std::size_t... Is>
      void scan_symmetry_sequence(std::istream& in, Symmetry& symmetry, std::index_sequence<Is...>) {
         (((Is == 0 ? in : (detail::ignore_until(in, ','), in)) >> std::get<Is>(symmetry)), ...);
      }
   } // namespace detail
   template<typename... T>
   std::ostream& operator<<(std::ostream& out, const Symmetry<T...>& symmetry) {
      using Symmetry = Symmetry<T...>;
      if constexpr (Symmetry::length != 0) {
         if constexpr (Symmetry::length == 1) {
            out << std::get<0>(symmetry);
         } else {
            out << '(';
            detail::print_symmetry_sequence(out, symmetry, typename Symmetry::index_sequence_t());
            out << ')';
         }
      }
      return out;
   }
   template<typename... T>
   std::istream& operator>>(std::istream& in, Symmetry<T...>& symmetry) {
      using Symmetry = Symmetry<T...>;
      if constexpr (Symmetry::length != 0) {
         if constexpr (Symmetry::length == 1) {
            in >> std::get<0>(symmetry);
         } else {
            detail::ignore_until(in, '(');
            detail::scan_symmetry_sequence(in, symmetry, typename Symmetry::index_sequence_t());
            detail::ignore_until(in, ')');
         }
      }
      return in;
   }

   /**
    * A type control console color
    */
   struct UnixColorCode {
      std::string color_code;
      UnixColorCode(const char* code) : color_code(code) {}
   };
   inline const UnixColorCode console_red = "\x1B[31m";
   inline const UnixColorCode console_green = "\x1B[32m";
   inline const UnixColorCode console_yellow = "\x1B[33m";
   inline const UnixColorCode console_blue = "\x1B[34m";
   inline const UnixColorCode console_origin = "\x1B[0m";
   inline std::ostream& operator<<(std::ostream& out, const UnixColorCode& value) {
      out << value.color_code;
      return out;
   }

   // tensor shape, text output only

   template<
         typename ScalarType,
         typename Symmetry,
         typename Name,
         typename = std::enable_if_t<is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>>>
   std::ostream& operator<<(std::ostream& out, const TensorShape<ScalarType, Symmetry, Name>& shape) {
      const auto& tensor = *shape.owner;
      out << '{' << console_green << "names" << console_origin << ':';
      out << tensor.names() << ',';
      out << console_green << "edges" << console_origin << ':';
      out << tensor.edges() << '}';
      return out;
   }

   // tensor text in out

   template<
         typename ScalarType,
         typename Symmetry,
         typename Name,
         typename = std::enable_if_t<is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>>>
   std::ostream& operator<<(std::ostream& out, const Tensor<ScalarType, Symmetry, Name>& tensor) {
      out << '{' << console_green << "names" << console_origin << ':';
      out << tensor.names() << ',';
      out << console_green << "edges" << console_origin << ':';
      out << tensor.edges() << ',';
      out << console_green << "blocks" << console_origin << ':';
      if constexpr (Symmetry::length == 0) {
         out << tensor.storage();
      } else {
         detail::print_list(
               out,
               [&tensor, it = tensor.blocks().begin()](std::ostream& out) mutable {
                  if (it.offset == 0) {
                     while (true) {
                        if (!it.valid) {
                           return true;
                        }
                        if (it->has_value()) {
                           break;
                        }
                        ++it;
                     }
                  }
                  std::vector<Symmetry> symmetries;
                  symmetries.reserve(tensor.rank());
                  for (auto j = 0; j < tensor.rank(); j++) {
                     symmetries.push_back(tensor.edges(j).segments(it.indices[j]).first);
                  }
                  out << console_yellow << symmetries << console_origin << ':';
                  detail::print_list(
                        out,
                        [offset = 0, l = it->value().data(), count = it->value().size()](std::ostream& out) mutable {
                           if (offset == count) {
                              return true;
                           }
                           if constexpr (is_complex<ScalarType>) {
                              detail::print_complex(out, l[offset]);
                           } else {
                              out << l[offset];
                           }
                           ++offset;
                           return offset == count;
                        },
                        '[',
                        ']');
                  while (true) {
                     ++it;
                     if (!it.valid) {
                        return true;
                     }
                     if (it->has_value()) {
                        return false;
                     }
                  }
               },
               '{',
               '}');
      }
      out << '}';
      return out;
   }

   template<
         typename ScalarType,
         typename Symmetry,
         typename Name,
         typename = std::enable_if_t<is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>>>
   std::istream& operator>>(std::istream& in, Tensor<ScalarType, Symmetry, Name>& tensor) {
      detail::ignore_until(in, ':');
      std::vector<Name> names;
      in >> names;
      detail::ignore_until(in, ':');
      std::vector<Edge<Symmetry>> edges;
      in >> edges;
      tensor = Tensor<ScalarType, Symmetry, Name>(std::move(names), std::move(edges));
      detail::ignore_until(in, ':');
      if constexpr (Symmetry::length == 0) {
         auto storage = tensor.storage().data();
         detail::scan_list(
               in,
               [offset = 0, l = storage](std::istream& in) mutable {
                  if constexpr (is_complex<ScalarType>) {
                     detail::scan_complex(in, l[offset++]);
                  } else {
                     in >> l[offset++];
                  }
               },
               '[',
               ']');
      } else {
         detail::scan_list(
               in,
               [&tensor](std::istream& in) {
                  std::vector<Symmetry> symmetries;
                  in >> symmetries;
                  detail::ignore_until(in, ':');
                  auto block = tensor.blocks(symmetries).data();
                  detail::scan_list(
                        in,
                        [offset = 0, l = block](std::istream& in) mutable {
                           if constexpr (is_complex<ScalarType>) {
                              detail::scan_complex(in, l[offset++]);
                           } else {
                              in >> l[offset++];
                           }
                        },
                        '[',
                        ']');
               },
               '{',
               '}');
      }
      detail::ignore_until(in, '}');
      return in;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::string Tensor<ScalarType, Symmetry, Name>::show() const {
      detail::basic_outstringstream<char> out;
      out << *this;
      return std::move(out).str();
   }

   // tensor bin out
   inline timer tensor_dump_guard("tensor_dump");

   template<
         typename ScalarType,
         typename Symmetry,
         typename Name,
         typename = std::enable_if_t<is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>>>
   std::ostream& operator<(std::ostream& out, const Tensor<ScalarType, Symmetry, Name>& tensor) {
      auto timer_guard = tensor_dump_guard();
      out << 'T' << 'A' << 'T';
      Rank version = 1;
      out < version;
      out < tensor.names();
      out < tensor.edges();
      out < tensor.storage();
      return out;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   std::string Tensor<ScalarType, Symmetry, Name>::dump() const {
      detail::basic_outstringstream<char> out;
      out < *this;
      return std::move(out).str();
   }

   // tensor bin in
   inline timer tensor_load_guard("tensor_load");

   template<
         typename ScalarType,
         typename Symmetry,
         typename Name,
         typename = std::enable_if_t<is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>>>
   std::istream& operator>(std::istream& in, Tensor<ScalarType, Symmetry, Name>& tensor) {
      auto timer_guard = tensor_load_guard();
      Rank version = 0;
      if (in.get() == 'T') {
         if (in.get() == 'A') {
            if (in.get() == 'T') {
               in > version;
            } else {
               in.unget();
               in.unget();
               in.unget();
               version = 0;
            }
         } else {
            in.unget();
            in.unget();
            version = 0;
         }
      } else {
         in.unget();
         version = 0;
      }
      if (version == 0) {
         std::vector<Name> names;
         in > names;
         std::vector<Edge<Symmetry>> edges;
         in > edges;
         tensor = Tensor<ScalarType, Symmetry, Name>(std::move(names), std::move(edges));
         in > tensor.storage();
         tensor._block_order_v0_to_v1();
      } else if (version == 1) {
         std::vector<Name> names;
         in > names;
         std::vector<Edge<Symmetry>> edges;
         in > edges;
         tensor = Tensor<ScalarType, Symmetry, Name>(std::move(names), std::move(edges));
         in > tensor.storage();
      }
      return in;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::load(const std::string& input) & {
      detail::basic_instringstream<char> in(input);
      in > *this;
      return *this;
   }

   // binary io move

   template<typename T>
   std::istream&& operator>(std::istream&& in, T& v) {
      in > v;
      return std::move(in);
   }
   template<typename T>
   std::ostream&& operator<(std::ostream&& out, const T& v) {
      out < v;
      return std::move(out);
   }
} // namespace TAT
#endif
