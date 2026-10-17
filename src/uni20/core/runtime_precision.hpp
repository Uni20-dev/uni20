#pragma once

#include "precision.hpp"
#include "scalar_traits.hpp"
#include <optional>

namespace uni20
{
/// \brief Whether numerical values of this scalar family carry runtime precision.
template <typename T> inline constexpr bool has_runtime_precision_v = false;
#if UNI20_ENABLE_MPFR
template <> inline constexpr bool has_runtime_precision_v<mpreal> = true;
#endif
#if UNI20_ENABLE_MPC
template <> inline constexpr bool has_runtime_precision_v<mpcomplex> = true;
#endif

namespace detail
{
template <class T> struct precision_component
{
    using type = T;
};
template <class T>
  requires is_complex_v<T>
struct precision_component<T>
{
    using type = make_real_t<T>;
};
template <class T> using precision_component_t = typename precision_component<std::remove_cvref_t<T>>::type;
} // namespace detail

/// \brief Concrete precision of a fixed scalar family, including complex components.
/// \details Exact scalar families report exact arithmetic. Runtime-precision
///          families need an instance and cannot use this type-only query.
template <class T>
  requires(!has_runtime_precision_v<std::remove_cvref_t<T>>) && requires {
    numeric_limits<detail::precision_component_t<T>>::digits();
    numeric_limits<detail::precision_component_t<T>>::is_exact;
  }
[[nodiscard]] constexpr Precision native_precision()
{
  using R = detail::precision_component_t<T>;
  if constexpr (numeric_limits<R>::is_exact)
    return Precision::exact();
  else
    return Precision::bits(numeric_limits<R>::digits());
}

/// \brief Read scalar precision without changing any scalar or provider state.
template <class T>
  requires requires(T const& value) { value.precision(); } || requires { native_precision<T>(); }
[[nodiscard]] constexpr Precision precision_of(T const& value)
{
  if constexpr (requires { value.precision(); })
    return value.precision();
  else
    return native_precision<T>();
}

/// \brief Reject an operation precision incompatible with a fixed scalar result.
/// \details Runtime scalars accept exact or finite precision; provider-specific
///          limits are checked by the provider before numerical work starts.
template <class T>
  requires(has_runtime_precision_v<std::remove_cvref_t<T>> || requires { native_precision<T>(); })
constexpr void require_precision(Precision p)
{
  if constexpr (!has_runtime_precision_v<std::remove_cvref_t<T>>)
    if (p != native_precision<T>()) throw std::invalid_argument("fixed-precision scalar requires its native precision");
}

/// \brief Construct or convert a scalar at one explicit precision.
template <class S, class V>
  requires((requires { native_precision<S>(); } && std::constructible_from<S, V const&>) ||
           (has_runtime_precision_v<S> && std::constructible_from<S, V const&, Precision>))
[[nodiscard]] constexpr S scalar_at_precision(V const& value, Precision p)
{
  require_precision<S>(p);
  if constexpr (has_runtime_precision_v<S>)
    return S(value, p);
  else
    return S(value);
}

namespace detail
{
// Empty for fixed-precision values; owners and views need no extra runtime state.
template <typename T, bool = has_runtime_precision_v<std::remove_cv_t<T>>> struct precision_default
{};
#if UNI20_ENABLE_MPFR
template <typename T> struct precision_default<T, true>
{
    std::optional<Precision> value = {};

    Precision get() const
    {
      if (!value) throw std::logic_error("tensor: no default working precision was supplied");
      return *value;
    }
    Precision default_precision() const { return this->get(); }
    void default_precision(Precision p) { value = p; }
    std::optional<Precision> default_precision_if_set() const noexcept { return value; }
};
#endif
} // namespace detail
} // namespace uni20
