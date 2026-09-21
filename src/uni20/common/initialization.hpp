#pragma once

#include "memory_diagnostics.hpp"
#include <uni20/core/scalar_traits.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace uni20
{

/// \brief Request storage whose numerical values must be written before being read.
struct uninitialized_t
{
    explicit constexpr uninitialized_t() = default;
};

/// \brief Explicit allocation tag for overwrite outputs and scratch space.
inline constexpr uninitialized_t uninitialized{};

/// \brief Numerical initialization requested from a storage implementation.
enum class StorageInitialization
{
  Zero,
  Uninitialized
};

/// \brief Opt a type into allocation without construction or destruction.
/// \details The default accepts trivially copyable and trivially destructible types. The standard
///          complex specializations are explicitly accepted as a practical extension: Uni20 relies
///          on their scalar-array representation and trivial lifetime behavior even on standard
///          libraries whose type traits do not yet report them as trivially copyable.
template <typename T>
inline constexpr bool enable_uninitialized_storage =
    std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

template <typename Real>
inline constexpr bool enable_uninitialized_storage<uni20::complex<Real>> =
    std::is_trivially_destructible_v<uni20::complex<Real>>;

/// \brief Whether raw allocation may establish storage for `T` without construction.
template <typename T>
concept uninitialized_ok = enable_uninitialized_storage<std::remove_cv_t<T>>;

namespace detail
{

template <typename T>
inline constexpr bool zero_initializable_v = [] {
  if constexpr (is_complex_v<T>)
  {
    using real_type = typename T::value_type;
    return requires { T{real_type{0}, real_type{0}}; };
  }
  else if constexpr (is_scalar_v<T>)
  {
    return requires { T{0}; };
  }
  else
  {
    return requires { T{}; };
  }
}();

template <typename T>
  requires zero_initializable_v<T>
T zero_value()
{
  if constexpr (is_complex_v<T>)
  {
    using real_type = typename T::value_type;
    return T{real_type{0}, real_type{0}};
  }
  else if constexpr (is_scalar_v<T>)
  {
    return T{0};
  }
  else
  {
    return T{};
  }
}

template <typename T>
inline constexpr bool has_signaling_nan_v = [] {
  if constexpr (is_complex_v<T>)
  {
    return uni20::numeric_limits<typename T::value_type>::has_signaling_NaN;
  }
  else
  {
    return is_real_v<T> && uni20::numeric_limits<T>::has_signaling_NaN;
  }
}();

template <typename T>
  requires has_signaling_nan_v<T>
T signaling_nan_value()
{
  if constexpr (is_complex_v<T>)
  {
    auto const value = uni20::numeric_limits<typename T::value_type>::signaling_NaN();
    return T{value, value};
  }
  else
  {
    return uni20::numeric_limits<T>::signaling_NaN();
  }
}

/// \brief Initialize live host elements, preserving construction requirements of nontrivial types.
/// \details For `uninitialized_ok` types the caller may supply raw allocated storage. Undefined-value
///          tracking is restricted to those types; it must not invalidate nontrivial object bookkeeping.
/// \pre Nonassignable elements must already have been value-constructed for the Zero policy.
/// \throws std::invalid_argument If Zero is requested for a type without a zero/default value.
template <typename T> void initialize_host_elements(T* data, std::size_t count, StorageInitialization initialization)
{
  if (count == 0) return;
  if (initialization == StorageInitialization::Zero)
  {
    if constexpr (!zero_initializable_v<T>)
    {
      throw std::invalid_argument("Zero initialization requires an element type with a zero/default value");
    }
    else if constexpr (std::is_assignable_v<T&, T>)
    {
      for (std::size_t i = 0; i < count; ++i)
        data[i] = zero_value<T>();
    }
    return;
  }
#if UNI20_FILL_UNINITIALIZED_SNAN
  if constexpr (uninitialized_ok<T> && has_signaling_nan_v<T>)
  {
    std::fill_n(data, count, signaling_nan_value<T>());
  }
#endif
  if constexpr (uninitialized_ok<T>)
  {
    memory_diagnostics::mark_uninitialized(data, count * sizeof(T));
  }
}

} // namespace detail
} // namespace uni20
