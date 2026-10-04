#pragma once

#include "numeric_limits.hpp"
#include "scalar_concepts.hpp"
#if UNI20_ENABLE_MPFR
#include "mpreal.hpp"
#endif
#if UNI20_ENABLE_MPC
#include "mpcomplex.hpp"
#endif
#include <cmath>
#include <complex>
#include <numeric>
#include <type_traits>
#include <utility>

namespace uni20
{

/**
 * \brief Scalar math helper utilities.
 *
 * \file math.hpp
 * \ingroup core
 */

/**
 * \brief Scalar helper utilities shared across Uni20 core algorithms.
 *
 * \defgroup core_math Scalar helper utilities
 * \ingroup core
 */

/// \brief A standard complex value is exact only when both components are exact.
template <class T> constexpr bool is_exact(detail::standard_complex<T> const& x)
{
  return uni20::is_exact(x.real()) && uni20::is_exact(x.imag());
}

/// \brief Indicates whether the Uni20 conjugation helper is a no-op for the provided scalar type.
/// \details Evaluates to `true` when the scalar is already real-valued or integral, allowing callers to skip
///         complex conjugation work. The variable template is `constexpr`, so the result may be used in
///         constant-expression contexts.
/// \tparam T Scalar type to inspect.
/// \ingroup core_math
template <typename T> inline constexpr bool has_trivial_conj = Real<T> || Integer<T>;

/// \brief Returns the complex conjugate for complex-valued scalars.
/// \details This overload forwards to `std::conj` and therefore returns a `uni20::complex<T>` copy of the
///         input value. It inherits the constexpr availability of `std::conj` (currently not `constexpr`).
/// \tparam T Component type of the complex scalar.
/// \param x Complex value whose conjugate is requested.
/// \return The complex conjugate of `x`.
/// \ingroup core_math
template <typename T> uni20::complex<T> conj(detail::standard_complex<T> x) { return std::conj(x); }

/// \brief Returns the conjugate of a real-valued scalar.
/// \details Real numbers are unchanged by conjugation, so the value is returned verbatim. The overload is
///         `constexpr`, enabling compile-time evaluation for literal arguments.
/// \tparam R Real scalar type.
/// \param x Real scalar to return.
/// \return `x`, unchanged.
/// \ingroup core_math
template <Real R> constexpr R conj(R const& x) { return x; }

/// \brief Returns the conjugate of an integer scalar.
/// \details Integer values are treated as reals for conjugation and therefore returned unchanged. The
///         overload is `constexpr`, enabling compile-time evaluation for literal arguments.
/// \tparam I Integer scalar type.
/// \param x Integer scalar to return.
/// \return `x`, unchanged.
/// \ingroup core_math
template <Integer I> constexpr I conj(I const& x) { return x; }

/// \brief Computes the Hermitian adjoint of a scalar value.
/// \details For scalar inputs the Hermitian adjoint is equivalent to the complex conjugate, so this helper
/// simply forwards to `conj`. When the selected `conj` overload is `constexpr`, this helper is as
/// well, preserving compile-time evaluation.
/// \tparam S Scalar type satisfying \c Scalar.
/// \param x Scalar value whose Hermitian adjoint is requested.
/// \return The result of calling `conj(x)`.
/// \ingroup core_math
template <Scalar S> constexpr auto herm(S x) { return uni20::conj(x); }

namespace detail
{
template <typename T, typename = void> struct numeric_limits_has_infinity : std::false_type
{};

template <typename T>
struct numeric_limits_has_infinity<
    T, std::void_t<decltype(uni20::numeric_limits<T>::has_infinity), decltype(uni20::numeric_limits<T>::infinity())>>
    : std::bool_constant<static_cast<bool>(uni20::numeric_limits<T>::has_infinity)>
{};

template <typename T>
inline constexpr bool numeric_limits_has_infinity_v = numeric_limits_has_infinity<std::remove_cvref_t<T>>::value;
} // namespace detail

/// \brief Returns whether an integer scalar is finite.
/// \details Integer Uni20 scalars have no NaN or infinity representation, so every value is finite.
/// \tparam I Integer scalar type.
/// \param x Integer scalar to inspect.
/// \return Always `true`.
/// \ingroup core_math
template <Integer I> constexpr bool isfinite(I const& x) noexcept
{
  (void)x;
  return true;
}

/// \brief Returns whether a real scalar is neither NaN nor positive/negative infinity.
/// \details This uses `uni20::numeric_limits<T>` so extension real scalar types can define their own infinity
///         representation without depending on standard-library overload coverage for `std::isfinite`.
///          Runtime-precision mpreal uses its native classification instead of type-only limits.
/// \tparam R Real scalar type.
/// \param x Real scalar to inspect.
/// \return `true` when `x` is finite.
/// \ingroup core_math
template <Real R> constexpr bool isfinite(R const& x)
{
  using value_type = std::remove_cvref_t<R>;
#if UNI20_ENABLE_MPFR
  if constexpr (std::same_as<value_type, mpreal>) return uni20::isfinite(x);
#endif
  if (!(x == x))
  {
    return false;
  }
  if constexpr (detail::numeric_limits_has_infinity_v<value_type>)
  {
    auto const infinity = uni20::numeric_limits<value_type>::infinity();
    return x != infinity && x != -infinity;
  }
  else
  {
    return true;
  }
}

/// \brief Returns whether both components of a complex scalar are finite.
/// \tparam T Real component type.
/// \param z Complex scalar to inspect.
/// \return `true` when both the real and imaginary components are finite.
/// \ingroup core_math
template <Complex C> constexpr bool isfinite(C const& z)
{
  return uni20::isfinite(z.real()) && uni20::isfinite(z.imag());
}

/// \brief Provides mutable access to the real component of a `uni20::complex` value.
/// \details This helper mirrors the `std::real` overload for lvalues while remaining `constexpr` and
/// `noexcept` for direct reference access.
/// \tparam T Component type of the complex scalar.
/// \param z Complex number whose real component will be exposed.
/// \return Reference to the real component of `z`.
/// \ingroup core_math
template <typename T> constexpr T& real(detail::standard_complex<T>& z) noexcept { return reinterpret_cast<T*>(&z)[0]; }

using std::real;

/// \brief Provides mutable access to the imaginary component of a `uni20::complex` value.
/// \details This helper mirrors the `std::imag` overload for lvalues while remaining `constexpr` and
/// `noexcept` for direct reference access.
/// \tparam T Component type of the complex scalar.
/// \param z Complex number whose imaginary component will be exposed.
/// \return Reference to the imaginary component of `z`.
/// \ingroup core_math
template <typename T> constexpr T& imag(detail::standard_complex<T>& z) noexcept { return reinterpret_cast<T*>(&z)[1]; }

using std::imag;

/// \brief Construct a scalar using an exemplar's type and numerical precision.
/// \details Fixed-precision types use ordinary construction. Runtime-precision
///          types use the exemplar's exact state or finite precision. This does
///          not invent working precision when the exemplar is exact.
template <Scalar S, class Value>
  requires(std::constructible_from<S, Value const&> ||
           requires(S const& exemplar, Value const& value) { S(value, exemplar.precision()); })
[[nodiscard]] auto scalar_like(S const& exemplar, Value const& value) -> S
{
  if constexpr (requires { S(value, exemplar.precision()); })
    return S(value, exemplar.precision());
  else
    return S(value);
}

namespace detail::scalar_math
{
template <class T>
concept native_argument = std::integral<T> || std::same_as<T, float> || std::same_as<T, double> ||
                          std::same_as<T, long double>
#if UNI20_HAS_FLOAT128
                          || std::same_as<T, uni20::float128>
#endif
    ;

// A missing extension overload must not be filled by a narrower native type.
// Integer promotion follows the selected standard operation.
template <class Result, class Arg> consteval bool preserves_real_precision()
{
  if constexpr (!Real<Arg>)
    return true;
  else if constexpr (Real<Result>)
    return numeric_limits<Result>::digits >= numeric_limits<Arg>::digits &&
           numeric_limits<Result>::max_exponent >= numeric_limits<Arg>::max_exponent;
  else
    return false;
}

// The deleted exact-match fallback rejects conversion-only classes, even when
// an associated namespace supplies standard floating overloads. Actual scalar
// overloads (including constrained templates and hidden friends) outrank it.
#define UNI20_SCALAR_MATH_ADL(NAME)                                                                            \
  namespace NAME##_adl                                                                                       \
  {                                                                                                         \
  template <class... Args> void NAME(Args const&...) = delete;                                                 \
  template <class... Args>                                                                                   \
  constexpr auto call(Args const&... args) noexcept(noexcept(NAME(args...))) -> decltype(NAME(args...))        \
  {                                                                                                         \
    return NAME(args...);                                                                                    \
  }                                                                                                         \
  }

#define UNI20_SCALAR_MATH_DISPATCH(NAME)                                                                      \
  UNI20_SCALAR_MATH_ADL(NAME)                                                                                 \
  struct NAME##_fn                                                                                           \
  {                                                                                                         \
      template <class... Args>                                                                               \
      constexpr auto operator()(Args const&... args) const                                                    \
          noexcept(noexcept(NAME##_adl::call(args...))) -> decltype(NAME##_adl::call(args...))                 \
      {                                                                                                     \
        return NAME##_adl::call(args...);                                                                     \
      }                                                                                                     \
      template <class... Args>                                                                               \
        requires((native_argument<Args> && ...) &&                                                           \
                 !requires(Args const&... args) { NAME##_adl::call(args...); })                                \
      constexpr auto operator()(Args const&... args) const                                                    \
          noexcept(noexcept(std::NAME(args...))) -> decltype(std::NAME(args...))                               \
        requires((preserves_real_precision<decltype(std::NAME(args...)), Args>()) && ...)                     \
      {                                                                                                     \
        return std::NAME(args...);                                                                            \
      }                                                                                                     \
  };

UNI20_SCALAR_MATH_DISPATCH(abs)
UNI20_SCALAR_MATH_DISPATCH(sqrt)
UNI20_SCALAR_MATH_DISPATCH(pow)
UNI20_SCALAR_MATH_DISPATCH(exp)
UNI20_SCALAR_MATH_DISPATCH(log)
UNI20_SCALAR_MATH_DISPATCH(log2)
UNI20_SCALAR_MATH_DISPATCH(sin)
UNI20_SCALAR_MATH_DISPATCH(cos)
UNI20_SCALAR_MATH_DISPATCH(ceil)
UNI20_SCALAR_MATH_DISPATCH(ldexp)
UNI20_SCALAR_MATH_DISPATCH(cbrt)
UNI20_SCALAR_MATH_DISPATCH(exp2)
UNI20_SCALAR_MATH_DISPATCH(expm1)
UNI20_SCALAR_MATH_DISPATCH(log10)
UNI20_SCALAR_MATH_DISPATCH(log1p)
UNI20_SCALAR_MATH_DISPATCH(asin)
UNI20_SCALAR_MATH_DISPATCH(acos)
UNI20_SCALAR_MATH_DISPATCH(sinh)
UNI20_SCALAR_MATH_DISPATCH(cosh)
UNI20_SCALAR_MATH_DISPATCH(tanh)
UNI20_SCALAR_MATH_DISPATCH(asinh)
UNI20_SCALAR_MATH_DISPATCH(acosh)
UNI20_SCALAR_MATH_DISPATCH(atanh)
UNI20_SCALAR_MATH_DISPATCH(tan)
UNI20_SCALAR_MATH_DISPATCH(atan)
UNI20_SCALAR_MATH_DISPATCH(atan2)
UNI20_SCALAR_MATH_DISPATCH(hypot)
#undef UNI20_SCALAR_MATH_DISPATCH

UNI20_SCALAR_MATH_ADL(real)
UNI20_SCALAR_MATH_ADL(imag)
UNI20_SCALAR_MATH_ADL(conj)
#undef UNI20_SCALAR_MATH_ADL

struct real_fn
{
    template <class T>
      requires(Real<T> || std::integral<T>)
    constexpr T operator()(T const& value) const
    {
      if constexpr (std::integral<T>) return value;
      else return uni20::conj(value);
    }
    template <class T>
      requires(!Real<T> && !std::integral<T>)
    constexpr auto operator()(T const& value) const
        noexcept(noexcept(std::remove_cvref_t<decltype(real_adl::call(value))>(real_adl::call(value))))
        -> std::remove_cvref_t<decltype(real_adl::call(value))>
      requires std::constructible_from<std::remove_cvref_t<decltype(real_adl::call(value))>,
                                       decltype(real_adl::call(value))>
    {
      return std::remove_cvref_t<decltype(real_adl::call(value))>(real_adl::call(value));
    }
};

struct imag_fn
{
    template <class T>
      requires(Real<T> || std::integral<T>)
    constexpr T operator()(T const& value) const
    {
      if constexpr (native_argument<T>) return T{};
      else return uni20::scalar_like(value, 0);
    }
    template <class T>
      requires(!Real<T> && !std::integral<T>)
    constexpr auto operator()(T const& value) const
        noexcept(noexcept(std::remove_cvref_t<decltype(imag_adl::call(value))>(imag_adl::call(value))))
        -> std::remove_cvref_t<decltype(imag_adl::call(value))>
      requires std::constructible_from<std::remove_cvref_t<decltype(imag_adl::call(value))>,
                                       decltype(imag_adl::call(value))>
    {
      return std::remove_cvref_t<decltype(imag_adl::call(value))>(imag_adl::call(value));
    }
};

struct conj_fn
{
    template <class T>
      requires(Real<T> || std::integral<T>)
    constexpr T operator()(T const& value) const
    {
      if constexpr (std::integral<T>) return value;
      else return uni20::conj(value);
    }
    template <class T>
      requires(!Real<T> && !std::integral<T>)
    constexpr auto operator()(T const& value) const
        noexcept(noexcept(conj_adl::call(value))) -> decltype(conj_adl::call(value))
    {
      return conj_adl::call(value);
    }
};
} // namespace detail::scalar_math

/// \brief Shared scalar math with native overloads and class-specific ADL customization.
/// \details Arguments are read through const references. Unsupported calls are constrained out;
///          class scalars cannot obtain support merely by converting to a native floating type.
///          Scalar overloads retain their exactness and precision rules, including trailing Precision
///          arguments where supported. This interface currently targets host execution.
namespace math
{
/// \brief Absolute value or complex magnitude, preserving the selected scalar implementation.
inline constexpr detail::scalar_math::abs_fn abs{};
/// \brief Square root; exact inputs requiring approximation need an explicit finite precision.
inline constexpr detail::scalar_math::sqrt_fn sqrt{};
/// \brief Power with the selected scalar's domain, exactness and precision rules.
inline constexpr detail::scalar_math::pow_fn pow{};
/// \brief Natural exponential with scalar-specific precision handling.
inline constexpr detail::scalar_math::exp_fn exp{};
/// \brief Natural logarithm with scalar-specific precision handling.
inline constexpr detail::scalar_math::log_fn log{};
/// \brief Base-two logarithm where supplied by the scalar implementation.
inline constexpr detail::scalar_math::log2_fn log2{};
/// \brief Sine with scalar-specific precision handling.
inline constexpr detail::scalar_math::sin_fn sin{};
/// \brief Cosine with scalar-specific precision handling.
inline constexpr detail::scalar_math::cos_fn cos{};
/// \brief Real cube root, including negative arguments.
inline constexpr detail::scalar_math::cbrt_fn cbrt{};
/// \brief Base-two exponential.
inline constexpr detail::scalar_math::exp2_fn exp2{};
/// \brief Natural exponential minus one, evaluated without subtractive cancellation.
inline constexpr detail::scalar_math::expm1_fn expm1{};
/// \brief Base-ten logarithm.
inline constexpr detail::scalar_math::log10_fn log10{};
/// \brief Natural logarithm of one plus the argument, retaining small increments.
inline constexpr detail::scalar_math::log1p_fn log1p{};
/// \brief Inverse sine in radians.
inline constexpr detail::scalar_math::asin_fn asin{};
/// \brief Inverse cosine in radians.
inline constexpr detail::scalar_math::acos_fn acos{};
/// \brief Hyperbolic sine.
inline constexpr detail::scalar_math::sinh_fn sinh{};
/// \brief Hyperbolic cosine.
inline constexpr detail::scalar_math::cosh_fn cosh{};
/// \brief Hyperbolic tangent.
inline constexpr detail::scalar_math::tanh_fn tanh{};
/// \brief Inverse hyperbolic sine.
inline constexpr detail::scalar_math::asinh_fn asinh{};
/// \brief Inverse hyperbolic cosine.
inline constexpr detail::scalar_math::acosh_fn acosh{};
/// \brief Inverse hyperbolic tangent.
inline constexpr detail::scalar_math::atanh_fn atanh{};
/// \brief Tangent in radians.
inline constexpr detail::scalar_math::tan_fn tan{};
/// \brief Inverse tangent in radians.
inline constexpr detail::scalar_math::atan_fn atan{};
/// \brief Quadrant-aware inverse tangent of y/x in radians.
inline constexpr detail::scalar_math::atan2_fn atan2{};
/// \brief Euclidean length with scalar-specific overflow and underflow handling.
inline constexpr detail::scalar_math::hypot_fn hypot{};
/// \brief Least integral value not less than the argument, in the scalar's result type.
inline constexpr detail::scalar_math::ceil_fn ceil{};
/// \brief Multiply by a power of two using the scalar's exponent interface.
inline constexpr detail::scalar_math::ldexp_fn ldexp{};
/// \brief Read the real component by value; real and integral scalars retain their type.
inline constexpr detail::scalar_math::real_fn real{};
/// \brief Read the imaginary component by value; real zero retains the input's type and precision.
inline constexpr detail::scalar_math::imag_fn imag{};
/// \brief Complex conjugation; real and integral values retain their type.
inline constexpr detail::scalar_math::conj_fn conj{};

/// \brief Square the magnitude, evaluating the scalar absolute value once.
/// \details This does not provide overflow scaling or extra working precision.
template <class T>
  requires requires(T const& value) { math::abs(value) * math::abs(value); }
auto abs_squared(T const& value)
{
  auto const magnitude = math::abs(value);
  return magnitude * magnitude;
}
} // namespace math

} // namespace uni20
