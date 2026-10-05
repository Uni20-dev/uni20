#pragma once

#include "types.hpp"
#include <numbers>

namespace uni20
{
namespace detail
{
template <class T> struct math_constants
{};

template <class T>
  requires(std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, long double>
#if UNI20_HAS_FLOAT128
           || std::same_as<T, float128>
#endif
  )
struct math_constants<T>
{
    static constexpr T pi = std::numbers::pi_v<T>;
    static constexpr T log_two = std::numbers::ln2_v<T>;
    static constexpr T euler_gamma = std::numbers::egamma_v<T>;
};
} // namespace detail

/// \brief Pi as a native constant or a precision-aware MPFR descriptor.
/// \details Native constants retain the requested type. With mpreal.hpp included,
///          pi<mpreal>.at(p) evaluates at finite precision p; arithmetic with an
///          approximate mpreal operand infers that operand's precision.
template <class Real>
  requires requires { detail::math_constants<Real>::pi; }
inline constexpr auto pi = detail::math_constants<Real>::pi;

/// \brief Natural logarithm of two, with the same type and precision rules as pi.
template <class Real>
  requires requires { detail::math_constants<Real>::log_two; }
inline constexpr auto log_two = detail::math_constants<Real>::log_two;

/// \brief Euler-Mascheroni constant, with the same type and precision rules as pi.
template <class Real>
  requires requires { detail::math_constants<Real>::euler_gamma; }
inline constexpr auto euler_gamma = detail::math_constants<Real>::euler_gamma;

/// \brief Precision-aware Catalan constant; currently available only for mpreal.
template <class Real>
  requires requires { detail::math_constants<Real>::catalan; }
inline constexpr auto catalan = detail::math_constants<Real>::catalan;
} // namespace uni20
