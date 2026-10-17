#include "mpfr.hpp"
#include "decompositions_impl.hpp"
#include "precision_scope.hpp"
#include <mplapack_mpfr.h>
#include <uni20/backend/lapack/lapack_error_presentation.hpp>
#include <uni20/common/trace.hpp>

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
using provider_scalar = std::conditional_t<std::same_as<Scalar, mpreal>, mpfrxx::mpfr_class, mpfrxx::mpc_class>;

void set(mpfrxx::mpfr_class& out, mpreal const& in) { in.copy_to(out.mpfr_data()); }
void set(mpfrxx::mpc_class& out, mpcomplex const& in) { in.copy_to(out.mpc_data()); }
mpreal get(mpfrxx::mpfr_class const& in, Precision p) { return mpreal(in.mpfr_data(), p); }
mpcomplex get(mpfrxx::mpc_class const& in, Precision p) { return mpcomplex(in.mpc_data(), p); }

template <class Scalar> auto pack(std::span<Scalar const> source)
{
  std::vector<provider_scalar<Scalar>> result(source.size());
  for (std::size_t i = 0; i < source.size(); ++i)
    set(result[i], source[i]);
  return result;
}
template <class Provider, class Scalar>
void unpack(std::vector<Provider> const& source, std::span<Scalar> out, Precision p)
{
  for (std::size_t i = 0; i < source.size(); ++i)
    out[i] = get(source[i], p);
}
void require_success(mplapackint info)
{
  if (info != 0) throw std::logic_error("MPLAPACK rejected validated adapter arguments");
}

template <class Scalar>
void gemm_impl(std::size_t m, std::size_t n, std::size_t k, Scalar const& alpha, std::span<Scalar const> a,
               std::span<Scalar const> b, Scalar const& beta, std::span<Scalar> c, Precision p)
{
  require_size(a.size(), m, k);
  require_size(b.size(), k, n);
  require_size(c.size(), m, n);
  auto pm = provider_integer(m), pn = provider_integer(n), pk = provider_integer(k);
  detail::precision_scope scope(p);
  provider_scalar<Scalar> pa, pb;
  set(pa, alpha);
  set(pb, beta);
  // Respect BLAS no-read semantics even for unset Uni20 elements.
  auto av = alpha == 0 || k == 0 ? std::vector<provider_scalar<Scalar>>(a.size()) : pack(a);
  auto bv = alpha == 0 || k == 0 ? std::vector<provider_scalar<Scalar>>(b.size()) : pack(b);
  auto cv = beta == 0 ? std::vector<provider_scalar<Scalar>>(c.size()) : pack(std::span<Scalar const>(c));
  if constexpr (std::same_as<Scalar, mpreal>)
    Rgemm("N", "N", pm, pn, pk, pa, av.data(), std::max<mplapackint>(1, pm), bv.data(), std::max<mplapackint>(1, pk),
          pb, cv.data(), std::max<mplapackint>(1, pm));
  else
    Cgemm("N", "N", pm, pn, pk, pa, av.data(), std::max<mplapackint>(1, pm), bv.data(), std::max<mplapackint>(1, pk),
          pb, cv.data(), std::max<mplapackint>(1, pm));
  unpack(cv, c, p);
}

template <class Scalar>
std::size_t getrf_impl(std::size_t n, std::span<Scalar> a, std::vector<std::size_t>& pivots, Precision p)
{
  require_size(a.size(), n, n);
  auto pn = provider_integer(n);
  detail::precision_scope scope(p);
  auto av = pack(std::span<Scalar const>(a));
  std::vector<mplapackint> pv(n);
  mplapackint info = 0;
  if constexpr (std::same_as<Scalar, mpreal>)
    Rgetrf(pn, pn, av.data(), std::max<mplapackint>(1, pn), pv.data(), info);
  else
    Cgetrf(pn, pn, av.data(), std::max<mplapackint>(1, pn), pv.data(), info);
  if (info < 0) require_success(info);
  unpack(av, a, p);
  pivots.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    pivots[i] = static_cast<std::size_t>(pv[i] - 1);
  return static_cast<std::size_t>(info);
}

template <class Scalar>
void getrs_impl(std::size_t n, std::size_t nrhs, std::span<Scalar const> a, std::span<std::size_t const> pivots,
                std::span<Scalar> b, Precision p)
{
  require_size(a.size(), n, n);
  require_size(b.size(), n, nrhs);
  if (pivots.size() != n) throw std::invalid_argument("MPLAPACK pivot count mismatch");
  auto pn = provider_integer(n), prhs = provider_integer(nrhs);
  detail::precision_scope scope(p);
  std::vector<mplapackint> pv(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    if (pivots[i] >= n) throw std::invalid_argument("MPLAPACK pivot outside matrix");
    pv[i] = provider_integer(pivots[i]) + 1;
  }
  auto av = pack(a), bv = pack(std::span<Scalar const>(b));
  mplapackint info = 0;
  if constexpr (std::same_as<Scalar, mpreal>)
    Rgetrs("N", pn, prhs, av.data(), std::max<mplapackint>(1, pn), pv.data(), bv.data(), std::max<mplapackint>(1, pn),
           info);
  else
    Cgetrs("N", pn, prhs, av.data(), std::max<mplapackint>(1, pn), pv.data(), bv.data(), std::max<mplapackint>(1, pn),
           info);
  require_success(info);
  unpack(bv, b, p);
}

// Query values are provider scalars, not Uni20 scalars. Convert directly from
// MPFR with checked integer bounds; never pass through a native floating type.
mpfr_srcptr query_real(mpfrxx::mpfr_class const& x) { return x.mpfr_data(); }
mpfr_srcptr query_real(mpfrxx::mpc_class const& x) { return mpc_realref(x.mpc_data()); }

template <class Provider, class Call> void with_workspace(char const* routine, Call&& call)
{
  Provider query;
  mplapackint info = 0;
  call(&query, mplapackint{-1}, info);
  require_success(info);
  auto value = query_real(query);
  if (!mpfr_number_p(value) || mpfr_cmp_ui(value, 1) < 0 || !mpfr_integer_p(value) ||
      !mpfr_fits_slong_p(value, MPFR_RNDN))
    throw std::length_error("invalid MPLAPACK workspace query");
  auto length = mpfr_get_si(value, MPFR_RNDN);
  if (!std::in_range<mplapackint>(length) || !std::in_range<std::size_t>(length))
    throw std::length_error("MPLAPACK workspace exceeds integer range");
  std::vector<Provider> work(static_cast<std::size_t>(length));
  call(work.data(), static_cast<mplapackint>(length), info);
  if (info < 0) require_success(info);
  if (info > 0)
  {
    CHECK(std::in_range<blas_int>(info), "MPLAPACK INFO exceeds LAPACK integer range");
    trace::raise(lapack::LapackError(routine, static_cast<blas_int>(info), "failed to converge"));
  }
}

struct DecompositionAdapter
{
    template <class S> using scalar = provider_scalar<S>;
    using real = mpfrxx::mpfr_class;
    using precision_scope = detail::precision_scope;
    static auto integer(std::size_t n) { return provider_integer(n); }
    static void check_size(std::size_t actual, std::size_t rows, std::size_t cols) { require_size(actual, rows, cols); }
    template <class S> static auto pack_values(std::span<S const> values) { return pack(values); }
    template <class P> static auto get_value(P const& value, Precision p) { return get(value, p); }
    template <class P, class S> static void unpack_values(std::vector<P> const& values, std::span<S> out, Precision p)
    {
      unpack(values, out, p);
    }
    template <class P, class Call> static void workspace(char const* routine, Call&& call)
    {
      with_workspace<P>(routine, std::forward<Call>(call));
    }
};

} // namespace

void gemm(std::size_t m, std::size_t n, std::size_t k, mpreal const& alpha, std::span<mpreal const> a,
          std::span<mpreal const> b, mpreal const& beta, std::span<mpreal> c, Precision p)
{
  gemm_impl(m, n, k, alpha, a, b, beta, c, p);
}
void gemm(std::size_t m, std::size_t n, std::size_t k, mpcomplex const& alpha, std::span<mpcomplex const> a,
          std::span<mpcomplex const> b, mpcomplex const& beta, std::span<mpcomplex> c, Precision p)
{
  gemm_impl(m, n, k, alpha, a, b, beta, c, p);
}
std::size_t getrf(std::size_t n, std::span<mpreal> a, std::vector<std::size_t>& pivots, Precision p)
{
  return getrf_impl(n, a, pivots, p);
}
std::size_t getrf(std::size_t n, std::span<mpcomplex> a, std::vector<std::size_t>& pivots, Precision p)
{
  return getrf_impl(n, a, pivots, p);
}
void getrs(std::size_t n, std::size_t nrhs, std::span<mpreal const> a, std::span<std::size_t const> pivots,
           std::span<mpreal> b, Precision p)
{
  getrs_impl(n, nrhs, a, pivots, b, p);
}
void getrs(std::size_t n, std::size_t nrhs, std::span<mpcomplex const> a, std::span<std::size_t const> pivots,
           std::span<mpcomplex> b, Precision p)
{
  getrs_impl(n, nrhs, a, pivots, b, p);
}
void qr(std::size_t m, std::size_t n, std::span<mpreal> a, std::span<mpreal> q, std::span<mpreal> triangular,
        Precision p)
{
  detail::orthogonal_factor_impl<DecompositionAdapter>(false, m, n, a, q, triangular, p);
}

void lq(std::size_t m, std::size_t n, std::span<mpreal> a, std::span<mpreal> q, std::span<mpreal> triangular,
        Precision p)
{
  detail::orthogonal_factor_impl<DecompositionAdapter>(true, m, n, a, q, triangular, p);
}

void eigh(std::size_t n, std::span<mpreal> a, std::span<mpreal> values, bool vectors, char triangle, Precision p)
{
  detail::eigh_impl<DecompositionAdapter>(n, a, values, vectors, triangle, p);
}

void svd(std::size_t m, std::size_t n, std::span<mpreal> a, std::span<mpreal> values, std::span<mpreal> u,
         std::span<mpreal> vh, char jobu, char jobvt, Precision p)
{
  detail::svd_impl<DecompositionAdapter>(m, n, a, values, u, vh, jobu, jobvt, p);
}

void eigen(std::size_t n, std::span<mpreal> a, std::span<mpcomplex> values, std::span<mpcomplex> vectors,
           bool compute_vectors, Precision p)
{
  detail::eigen_impl<DecompositionAdapter>(n, a, values, vectors, compute_vectors, p);
}

void qr(std::size_t m, std::size_t n, std::span<mpcomplex> a, std::span<mpcomplex> q, std::span<mpcomplex> triangular,
        Precision p)
{
  detail::orthogonal_factor_impl<DecompositionAdapter>(false, m, n, a, q, triangular, p);
}

void lq(std::size_t m, std::size_t n, std::span<mpcomplex> a, std::span<mpcomplex> q, std::span<mpcomplex> triangular,
        Precision p)
{
  detail::orthogonal_factor_impl<DecompositionAdapter>(true, m, n, a, q, triangular, p);
}

void eigh(std::size_t n, std::span<mpcomplex> a, std::span<mpreal> values, bool vectors, char triangle, Precision p)
{
  detail::eigh_impl<DecompositionAdapter>(n, a, values, vectors, triangle, p);
}

void svd(std::size_t m, std::size_t n, std::span<mpcomplex> a, std::span<mpreal> values, std::span<mpcomplex> u,
         std::span<mpcomplex> vh, char jobu, char jobvt, Precision p)
{
  detail::svd_impl<DecompositionAdapter>(m, n, a, values, u, vh, jobu, jobvt, p);
}

void eigen(std::size_t n, std::span<mpcomplex> a, std::span<mpcomplex> values, std::span<mpcomplex> vectors,
           bool compute_vectors, Precision p)
{
  detail::eigen_impl<DecompositionAdapter>(n, a, values, vectors, compute_vectors, p);
}
} // namespace uni20::mplapack
