#pragma once

#include <uni20/core/types.hpp>
#include <uni20/random/transform_group.hpp>

#ifndef __CUDA_ARCH__
#include <uni20/core/math.hpp>
#endif

#include <cmath>
#include <concepts>
#include <numbers>

namespace uni20::random
{
namespace detail
{
template <class Real> UNI20_HOST_DEVICE ValueBlock<Real, 2> standard_normal_pair(Real u1, Real u2)
{
  Real const x = 2 * u2;
#ifdef __CUDA_ARCH__
  // Double is the device normal precision supported by the current backend.
  static_assert(std::same_as<Real, double>);
  double const radius = std::sqrt(-2.0 * std::log(u1));
  double sine, cosine;
  ::sincospi(x, &sine, &cosine);
#else
  Real const radius = math::sqrt(-2 * math::log(u1));
  Real const sine = math::sinpi(x);
  Real const cosine = math::cospi(x);
#endif
  return {{radius * cosine, radius * sine}};
}
} // namespace detail

/// \brief Transform unit uniforms to values between low and high.
/// \pre The endpoints are finite and ordered, and their difference is representable.
template <class T>
  requires requires { UnitUniform<T>::word_count; }
struct Uniform
{
    using group_type = TransformGroup<T, ValueBlock<T, 1>, UnitInterval::closed_open>;
    using input_type = typename group_type::input_type;
    using result_type = typename group_type::result_type;
    T low = 0;
    T high = 1;

    UNI20_HOST_DEVICE constexpr ValueBlock<T, 1> from_uniforms(T u) const { return {{low + (high - low) * u}}; }
    UNI20_HOST_DEVICE constexpr result_type operator()(input_type const& group) const
    {
      return group_type::apply(group, *this);
    }
};

namespace detail
{
/// \brief Transform two unit uniforms into two real normals with mean mu and standard deviation sigma.
/// \pre sigma is finite and nonnegative.
template <class Real> struct RealNormalDistribution
{
    using group_type = TransformGroup<Real, ValueBlock<Real, 2>, UnitInterval::open_closed, UnitInterval::closed_open>;
    using input_type = group_type::input_type;
    using result_type = group_type::result_type;
    Real mu = 0;
    Real sigma = 1;

    UNI20_HOST_DEVICE ValueBlock<Real, 2> from_uniforms(Real u1, Real u2) const
    {
      auto pair = detail::standard_normal_pair(u1, u2);
      for (auto& x : pair.values)
        x = mu + sigma * x;
      return pair;
    }
    UNI20_HOST_DEVICE result_type operator()(input_type const& group) const { return group_type::apply(group, *this); }
};

/// \brief Transform two unit uniforms into one circular complex normal with E|z - mu|² = sigma².
/// \pre sigma is finite and nonnegative.
template <class Real> struct ComplexNormalDistribution
{
    using group_type =
        TransformGroup<Real, ValueBlock<complex<Real>, 1>, UnitInterval::open_closed, UnitInterval::closed_open>;
    using input_type = group_type::input_type;
    using result_type = group_type::result_type;
    complex<Real> mu = {};
    Real sigma = 1;

    UNI20_HOST_DEVICE ValueBlock<complex<Real>, 1> from_uniforms(Real u1, Real u2) const
    {
      auto const pair = detail::standard_normal_pair(u1, u2);
      Real const scale = sigma * (std::numbers::sqrt2_v<Real> / 2);
      return {{complex<Real>{mu.real() + scale * pair.values[0], mu.imag() + scale * pair.values[1]}}};
    }
    UNI20_HOST_DEVICE result_type operator()(input_type const& group) const { return group_type::apply(group, *this); }
};
} // namespace detail

/// \brief Two real Box–Muller values per Philox block at double precision.
using Normal = detail::RealNormalDistribution<double>;
/// \brief One circular complex Box–Muller value per Philox block at double precision.
using ComplexNormal = detail::ComplexNormalDistribution<double>;

} // namespace uni20::random
