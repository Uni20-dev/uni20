#include "binary80.hpp"
#include <mpblas_binary80.h>
#include <mplapack_binary80.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20::mplapack
{
namespace
{
mplapackint provider_integer(std::size_t n)
{
  if (!std::in_range<mplapackint>(n)) throw std::length_error("MPLAPACK dimension exceeds provider integer range");
  return static_cast<mplapackint>(n);
}

void require_size(std::size_t actual, std::size_t rows, std::size_t cols)
{
  if (cols && rows > std::numeric_limits<std::size_t>::max() / cols)
    throw std::length_error("MPLAPACK packed matrix size overflow");
  if (actual != rows * cols) throw std::invalid_argument("MPLAPACK packed matrix size mismatch");
}

template <class Scalar>
using provider_scalar =
    std::conditional_t<std::same_as<Scalar, float80>, mplapack_binary80_t, uni20::complex<mplapack_binary80_t>>;

mplapack_binary80_t convert(float80 value) { return static_cast<mplapack_binary80_t>(value); }
uni20::complex<mplapack_binary80_t> convert(complex160 const& value)
{
  return {convert(value.real()), convert(value.imag())};
}
float80 get(mplapack_binary80_t const& value) { return static_cast<float80>(value); }
complex160 get(uni20::complex<mplapack_binary80_t> const& value)
{
  return {static_cast<float80>(value.real()), static_cast<float80>(value.imag())};
}

template <class Scalar> auto pack(std::span<Scalar const> source)
{
  std::vector<provider_scalar<Scalar>> result(source.size());
  for (std::size_t i = 0; i < source.size(); ++i)
    result[i] = convert(source[i]);
  return result;
}
template <class Provider, class Scalar> void unpack(std::vector<Provider> const& source, std::span<Scalar> out)
{
  for (std::size_t i = 0; i < source.size(); ++i)
    out[i] = get(source[i]);
}
void require_success(mplapackint info)
{
  if (info != 0) throw std::logic_error("MPLAPACK rejected validated adapter arguments");
}

template <class Scalar>
void gemm_impl(std::size_t m, std::size_t n, std::size_t k, Scalar const& alpha, std::span<Scalar const> a,
               std::span<Scalar const> b, Scalar const& beta, std::span<Scalar> c)
{
  require_size(a.size(), m, k);
  require_size(b.size(), k, n);
  require_size(c.size(), m, n);
  auto pm = provider_integer(m), pn = provider_integer(n), pk = provider_integer(k);
  provider_scalar<Scalar> pa, pb;
  pa = convert(alpha);
  pb = convert(beta);
  // Respect BLAS no-read semantics without reading unused operands.
  auto av = alpha == Scalar{0} || k == 0 ? std::vector<provider_scalar<Scalar>>(a.size()) : pack(a);
  auto bv = alpha == Scalar{0} || k == 0 ? std::vector<provider_scalar<Scalar>>(b.size()) : pack(b);
  auto cv = beta == Scalar{0} ? std::vector<provider_scalar<Scalar>>(c.size()) : pack(std::span<Scalar const>(c));
  if constexpr (std::same_as<Scalar, float80>)
    Rgemm("N", "N", pm, pn, pk, pa, av.data(), std::max<mplapackint>(1, pm), bv.data(), std::max<mplapackint>(1, pk),
          pb, cv.data(), std::max<mplapackint>(1, pm));
  else
    Cgemm("N", "N", pm, pn, pk, pa, av.data(), std::max<mplapackint>(1, pm), bv.data(), std::max<mplapackint>(1, pk),
          pb, cv.data(), std::max<mplapackint>(1, pm));
  unpack(cv, c);
}

template <class Scalar> std::size_t getrf_impl(std::size_t n, std::span<Scalar> a, std::vector<std::size_t>& pivots)
{
  require_size(a.size(), n, n);
  auto pn = provider_integer(n);
  auto av = pack(std::span<Scalar const>(a));
  std::vector<mplapackint> pv(n);
  mplapackint info = 0;
  if constexpr (std::same_as<Scalar, float80>)
    Rgetrf(pn, pn, av.data(), std::max<mplapackint>(1, pn), pv.data(), info);
  else
    Cgetrf(pn, pn, av.data(), std::max<mplapackint>(1, pn), pv.data(), info);
  if (info < 0) require_success(info);
  unpack(av, a);
  pivots.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    pivots[i] = static_cast<std::size_t>(pv[i] - 1);
  return static_cast<std::size_t>(info);
}

template <class Scalar>
void getrs_impl(std::size_t n, std::size_t nrhs, std::span<Scalar const> a, std::span<std::size_t const> pivots,
                std::span<Scalar> b)
{
  require_size(a.size(), n, n);
  require_size(b.size(), n, nrhs);
  if (pivots.size() != n) throw std::invalid_argument("MPLAPACK pivot count mismatch");
  auto pn = provider_integer(n), prhs = provider_integer(nrhs);
  std::vector<mplapackint> pv(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    if (pivots[i] >= n) throw std::invalid_argument("MPLAPACK pivot outside matrix");
    pv[i] = provider_integer(pivots[i]) + 1;
  }
  auto av = pack(a), bv = pack(std::span<Scalar const>(b));
  mplapackint info = 0;
  if constexpr (std::same_as<Scalar, float80>)
    Rgetrs("N", pn, prhs, av.data(), std::max<mplapackint>(1, pn), pv.data(), bv.data(), std::max<mplapackint>(1, pn),
           info);
  else
    Cgetrs("N", pn, prhs, av.data(), std::max<mplapackint>(1, pn), pv.data(), bv.data(), std::max<mplapackint>(1, pn),
           info);
  require_success(info);
  unpack(bv, b);
}
} // namespace

void gemm(std::size_t m, std::size_t n, std::size_t k, float80 const& alpha, std::span<float80 const> a,
          std::span<float80 const> b, float80 const& beta, std::span<float80> c)
{
  gemm_impl(m, n, k, alpha, a, b, beta, c);
}
void gemm(std::size_t m, std::size_t n, std::size_t k, complex160 const& alpha, std::span<complex160 const> a,
          std::span<complex160 const> b, complex160 const& beta, std::span<complex160> c)
{
  gemm_impl(m, n, k, alpha, a, b, beta, c);
}
std::size_t getrf(std::size_t n, std::span<float80> a, std::vector<std::size_t>& pivots)
{
  return getrf_impl(n, a, pivots);
}
std::size_t getrf(std::size_t n, std::span<complex160> a, std::vector<std::size_t>& pivots)
{
  return getrf_impl(n, a, pivots);
}
void getrs(std::size_t n, std::size_t nrhs, std::span<float80 const> a, std::span<std::size_t const> pivots,
           std::span<float80> b)
{
  getrs_impl(n, nrhs, a, pivots, b);
}
void getrs(std::size_t n, std::size_t nrhs, std::span<complex160 const> a, std::span<std::size_t const> pivots,
           std::span<complex160> b)
{
  getrs_impl(n, nrhs, a, pivots, b);
}
} // namespace uni20::mplapack
