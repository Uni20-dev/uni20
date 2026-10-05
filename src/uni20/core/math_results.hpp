#pragma once
#include <cstdint>

namespace uni20
{
/// \brief A binary significand and exponent: value = fraction * 2^exponent.
template <class Real> struct frexp_result
{
    using value_type = Real;
    Real fraction;
    std::int64_t exponent;
};
/// \brief Signed fractional and integral parts of a real value.
template <class Real> struct modf_result
{
    using value_type = Real;
    Real fraction;
    Real integer;
};
/// \brief Nearest-even remainder and the signed low three bits of its quotient.
template <class Real> struct remquo_result
{
    using value_type = Real;
    Real remainder;
    int quotient;
};
/// \brief Sine and cosine evaluated at one input and precision.
template <class Real> struct sincos_result
{
    using value_type = Real;
    Real sin;
    Real cos;
};
/// \brief Hyperbolic sine and cosine evaluated at one input and precision.
template <class Real> struct sinhcosh_result
{
    using value_type = Real;
    Real sinh;
    Real cosh;
};
/// \brief Logarithm of the absolute Gamma value and Gamma's sign.
template <class Real> struct lgamma_result
{
    using value_type = Real;
    Real value;
    int sign;
};
} // namespace uni20
