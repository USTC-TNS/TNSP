module;

#include <array>
#include <variant>
#include <vector>

export module TAT.multidimension_span;

import TAT.type_alias;
import TAT.timer;
import TAT.compile_information;
import TAT.log_and_exception;
import TAT.const_integral;

namespace TAT {
   template<typename T, typename U = std::vector<Size>>
   class mdspan {
      // data is const when
      // const mdspan, const T
      // const mdspan, T
      // mdspan, const T
      // data is not const only when
      // mdspan, T

      using vector_t = U; // used to present indices
      using self_t = mdspan<T, U>;

    public:
      template<bool is_const>
      struct iterator_general;
      using iterator = iterator_general<false>;
      using const_iterator = iterator_general<true>;
      friend const_iterator;
      friend iterator;

    private:
      T* m_data;
      const Rank m_rank;
      const Size m_size;
      const vector_t m_leadings;
      const vector_t m_dimensions;

    public:
      Rank rank() const {
         return m_rank;
      }
      Size size() const {
         return m_size;
      }
      const vector_t& leadings() const {
         return m_leadings;
      }
      const vector_t& dimensions() const {
         return m_dimensions;
      }
      Size leadings(Rank i) const {
         return m_leadings[i];
      }
      Size dimensions(Rank i) const {
         return m_dimensions[i];
      }
      T* data() {
         return m_data;
      }
      const T* data() const {
         return m_data;
      }
      void set_data(T* pointer) {
         m_data = pointer;
      }

    private:
      static vector_t contiguous_leadings(const vector_t& dimension) {
         auto rank = dimension.size();
         auto result = vector_t(rank);
         for (auto i = rank; i-- > 0;) {
            if (i == rank - 1) {
               result[i] = 1;
            } else {
               result[i] = result[i + 1] * dimension[i + 1];
            }
         }
         return result;
      }
      static Size get_size(const vector_t& dimension) {
         Size size = 1;
         for (const auto& i : dimension) {
            size *= i;
         }
         return size;
      }

    public:
      mdspan(T* pointer, vector_t input_dimensions, vector_t input_leadings) :
            m_data(pointer),
            m_rank(input_dimensions.size()),
            m_size(get_size(input_dimensions)),
            m_leadings(std::move(input_leadings)),
            m_dimensions(std::move(input_dimensions)) {}
      mdspan(T* pointer, vector_t ds) :
            m_data(pointer),
            m_rank(ds.size()),
            m_size(get_size(ds)),
            m_leadings(contiguous_leadings(ds)),
            m_dimensions(std::move(ds)) {}

    private:
      template<typename Vector>
      Size get_offset(const Vector& indices) const {
         Size offset = 0;
         for (auto i = 0; i < rank(); i++) {
            offset += indices[i] * leadings(i);
         }
         return offset;
      }
    public:
      template<typename Vector>
      const T& at(const Vector& indices) const {
         return data()[get_offset(indices)];
      }
      template<typename Vector>
      T& at(const Vector& indices) {
         return data()[get_offset(indices)];
      }

    public:
      template<bool is_const>
      struct iterator_general {
       private:
         template<typename Y>
         using maybe_const = std::conditional_t<is_const, const Y, Y>;

       public:
         maybe_const<self_t>* owner;
         Size offset;
         bool valid;
         vector_t indices;

         iterator_general(maybe_const<self_t>* owner, Size offset, bool valid, vector_t indices) :
               owner(owner),
               offset(offset),
               valid(valid),
               indices(std::move(indices)) {}

         maybe_const<T>& operator*() const {
            return owner->data()[offset];
         }
         maybe_const<T>* operator->() const {
            return &owner->data()[offset];
         }

         iterator_general& operator++() {
            Rank current = owner->rank();
            while (true) {
               if (current-- == 0) {
                  valid = false;
                  return *this;
               }
               Size dimension = owner->dimensions(current);
               Size leading = owner->leadings(current);
               offset += leading;
               ++indices[current];
               if (indices[current] != dimension) {
                  break;
               }
               indices[current] = 0;
               offset -= dimension * leading;
            }
            return *this;
         }

         bool operator==(const iterator_general<is_const>& other) const {
            return (owner == other.owner) && ((!valid && !other.valid) || (valid && other.valid && offset == other.offset));
         }
         bool operator!=(const iterator_general<is_const>& other) const {
            return !(*this == other);
         }
      };

    public:
      const_iterator begin() const {
         if (size() != 0) {
            return const_iterator(this, 0, true, vector_t(rank(), 0));
         } else {
            return const_iterator(this, 0, false, vector_t(rank(), 0));
         }
      }

      const_iterator end() const {
         return const_iterator(this, 0, false, vector_t(rank(), 0));
      }

      iterator begin() {
         if (size() != 0) {
            return iterator(this, 0, true, vector_t(rank(), 0));
         } else {
            return iterator(this, 0, false, vector_t(rank(), 0));
         }
      }

      iterator end() {
         return iterator(this, 0, false, vector_t(rank(), 0));
      }

    public:
      template<typename Vector>
      mdspan<T, U> transpose(const Vector& plan) {
         vector_t new_dimensions(rank());
         vector_t new_leadings(rank());
         for (auto i = 0; i < rank(); i++) {
            new_dimensions[i] = dimensions(plan[i]);
            new_leadings[i] = leadings(plan[i]);
         }
         return mdspan<T, U>(data(), std::move(new_dimensions), std::move(new_leadings));
      }
      template<typename Vector>
      mdspan<const T, U> transpose(const Vector& plan) const {
         vector_t new_dimensions(rank());
         vector_t new_leadings(rank());
         for (auto i = 0; i < rank(); i++) {
            new_dimensions[i] = dimensions(plan[i]);
            new_leadings[i] = leadings(plan[i]);
         }
         return mdspan<const T, U>(data(), std::move(new_dimensions), std::move(new_leadings));
      }
   };

   inline timer transform_kernel_guard("transform_kernel");

   namespace detail {
      template<typename T1, typename T2, typename Vector, typename Func, typename LastSize = int>
      void mdspan_transform_kernel(
            const T1* __restrict source,
            T2* __restrict destination,
            const Vector& dimensions,
            const Vector& leadings_source,
            const Vector& leadings_destination,
            Func&& func,
            const LastSize last_size = 0) {
         auto kernel_guard = transform_kernel_guard();
         constexpr bool loop_last = !std::is_same_v<LastSize, int>; // not int -> last_size specified.
         const Rank rank = dimensions.size();
         auto indices = Vector(rank, 0);
         // if size = 0, program won't call this function
         while (true) {
            if constexpr (loop_last) {
               for (auto i = 0; i < last_size.value(); i++) {
                  *destination++ = func(*source++);
               }
               indices.back() = dimensions.back() - 1;
               --source;
               --destination;
            } else {
               *destination = func(*source);
            }
            Rank current = rank;
            while (true) {
               if (current-- == 0) {
                  return;
               }
               Size dimension = dimensions[current];
               Size leading_source = leadings_source[current];
               Size leading_destination = leadings_destination[current];
               source += leading_source;
               destination += leading_destination;
               ++indices[current];
               if (indices[current] != dimension) {
                  break;
               }
               indices[current] = 0;
               source -= leading_source * dimension;
               destination -= leading_destination * dimension;
            }
         }
      }
   } // namespace detail
   template<typename T1, typename T2, typename U1, typename U2, typename Func>
   void mdspan_transform(const mdspan<T1, U1>& source, mdspan<T2, U2>& destination, Func&& func) {
      if constexpr (debug_mode) {
         if (!std::equal(source.dimensions().begin(), source.dimensions().end(), destination.dimensions().begin(), destination.dimensions().end())) {
            error("transform data between mdspan where dimension are not the same");
         }
      }
      if (destination.size() == 0) {
         return;
      }
      if (destination.size() == 1) {
         *destination.data() = func(*source.data());
         return;
      }

      auto rank = destination.rank();
      auto result_dimensions = U2();
      auto result_leadings_destination = U2();
      auto result_leadings_source = U2();
      result_dimensions.reserve(rank);
      result_leadings_destination.reserve(rank);
      result_leadings_source.reserve(rank);
      // Remove all dimension=1 edge and squash leading squashable edges.
      // There is at least one non-1 dimension edge
      Size dimension = 0;
      Size leading_destination = 0;
      Size leading_source = 0;
      for (auto current = 0; current < rank; current++) {
         auto dimension_current = destination.dimensions(current);
         auto leading_destination_current = destination.leadings(current);
         auto leading_source_current = source.leadings(current);
         if (dimension_current == 1) {
            continue;
         }
         if ((leading_destination == dimension_current * leading_destination_current) &&
             (leading_source == dimension_current * leading_source_current)) {
            dimension *= dimension_current;
            leading_destination = leading_destination_current;
            leading_source = leading_source_current;
            continue;
         }
         if (dimension != 0) {
            result_dimensions.push_back(dimension);
            result_leadings_destination.push_back(leading_destination);
            result_leadings_source.push_back(leading_source);
         }
         dimension = dimension_current;
         leading_destination = leading_destination_current;
         leading_source = leading_source_current;
      }
      result_dimensions.push_back(dimension);
      result_leadings_destination.push_back(leading_destination);
      result_leadings_source.push_back(leading_source);

      if (result_leadings_destination.back() == 1 && result_leadings_source.back() == 1) {
         Size line_size = result_dimensions.back();
         std::visit(
               [&](const auto& const_line_size) {
                  detail::mdspan_transform_kernel(
                        source.data(),
                        destination.data(),
                        result_dimensions,
                        result_leadings_source,
                        result_leadings_destination,
                        std::forward<Func>(func),
                        const_line_size);
               },
               to_const_integral_0_to_16<Size>(line_size));
      } else {
         detail::mdspan_transform_kernel(
               source.data(),
               destination.data(),
               result_dimensions,
               result_leadings_source,
               result_leadings_destination,
               std::forward<Func>(func));
      }
   }

   template<typename Vector, typename T>
   void matrix_transpose(Size m, Size n, const T* source, T* destination) {
      // source m*n
      // destination n*m
      detail::mdspan_transform_kernel(source, destination, Vector{n, m}, Vector{1, n}, Vector{m, 1}, [](const auto& x) {
         return x;
      });
   }
} // namespace TAT
