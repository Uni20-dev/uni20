#pragma once

#include <limits>
#include <type_traits>

namespace uni20
{

/// \brief Project-level numeric limits customization point.
/// \details The primary template inherits `std::numeric_limits<T>` so ordinary arithmetic types use the standard
///          library implementation. Uni20 scalar backends should specialize this template, not `std::numeric_limits`,
///          for extension or library scalar types whose standard-library limits are missing or incomplete.
/// \tparam T Scalar type to inspect.
/// \ingroup core_math
template <typename T> struct numeric_limits : std::numeric_limits<T>
{};

template <typename T> struct numeric_limits<T const> : numeric_limits<T>
{};

template <typename T> struct numeric_limits<T volatile> : numeric_limits<T>
{};

template <typename T> struct numeric_limits<T const volatile> : numeric_limits<T>
{};

/// \brief Query a value's exact-arithmetic state, or its type's static exactness.
/// \details Runtime scalar types provide an is_exact() member. For ordinary
///          scalars this returns numeric_limits<T>::is_exact: integers are exact,
///          floating types are approximate even when their value is an integer.
///          Unset runtime scalar values report false.
template <class T>
  requires(requires(T const& x) { x.is_exact(); } || numeric_limits<T>::is_specialized)
constexpr bool is_exact(T const& x)
{
  if constexpr (requires { x.is_exact(); })
    return x.is_exact();
  else
    return numeric_limits<T>::is_exact;
}

/// \brief True when Uni20 has numeric limits for `T`.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T>
inline constexpr bool has_numeric_limits_v = numeric_limits<std::remove_cvref_t<T>>::is_specialized;

} // namespace uni20
