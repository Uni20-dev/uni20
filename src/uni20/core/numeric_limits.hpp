#pragma once

#include <limits>
#include <type_traits>

namespace uni20
{

/// \brief Project-level numeric limits customization point.
/// \details The primary template is intentionally undefined: unsupported types
///          have no fallback values. Native arithmetic types explicitly delegate
///          to std::numeric_limits. Library scalars must specialize this template.
/// \tparam T Scalar type to inspect.
/// \ingroup core_math
template <typename T> struct numeric_limits;

template <typename T>
  requires(std::is_arithmetic_v<T> && std::numeric_limits<T>::is_specialized)
struct numeric_limits<T> : std::numeric_limits<T>
{
    using std::numeric_limits<T>::epsilon;

    /// \brief Spacing above one, accepting an exemplar for scalar-generic code.
    template <class U>
      requires(std::is_same_v<T, U> && std::numeric_limits<T>::is_specialized)
    static constexpr T epsilon(U const&) noexcept
    {
      return std::numeric_limits<T>::epsilon();
    }
};

template <typename T>
  requires requires { numeric_limits<T>::is_specialized; }
struct numeric_limits<T const> : numeric_limits<T>
{};

template <typename T>
  requires requires { numeric_limits<T>::is_specialized; }
struct numeric_limits<T volatile> : numeric_limits<T>
{};

template <typename T>
  requires requires { numeric_limits<T>::is_specialized; }
struct numeric_limits<T const volatile> : numeric_limits<T>
{};

/// \brief True when Uni20 has numeric limits for `T`.
/// \details Runtime-precision specializations require a value or precision for
///          precision-dependent queries; this trait does not promise static limits.
/// \tparam T Type to inspect.
/// \ingroup core_math
template <typename T>
inline constexpr bool has_numeric_limits_v =
    requires { requires numeric_limits<std::remove_cvref_t<T>>::is_specialized; };

/// \brief Query a value's exact-arithmetic state, or its type's static exactness.
/// \details Runtime scalar types provide an is_exact() member. For ordinary
///          scalars this returns numeric_limits<T>::is_exact: integers are exact,
///          floating types are approximate even when their value is an integer.
///          Unset runtime scalar values report false.
template <class T>
  requires(requires(T const& x) { x.is_exact(); } || has_numeric_limits_v<T>)
constexpr bool is_exact(T const& x)
{
  if constexpr (requires { x.is_exact(); })
    return x.is_exact();
  else
    return numeric_limits<T>::is_exact;
}

} // namespace uni20
