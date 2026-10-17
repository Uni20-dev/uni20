#pragma once

/**
 * \file decompositions_impl.hpp
 * \brief Shared packed MPLAPACK decomposition algorithms for compiled scalar adapters.
 * \details Compiled provider targets select their declarations through MPLAPACK
 *          build definitions. Adapter owns conversion, workspace-query lowering
 *          and precision context; public Uni20 headers never expose provider types.
 */
#if defined(MPLAPACK_BUILD_WITH_MPFR)
#include <mplapack_mpfr.h>
#endif
#if defined(MPLAPACK_BUILD_WITH_BINARY80)
#include <mplapack_binary80.h>
#endif

#include <algorithm>
#include <limits>
#include <span>
#include <uni20/common/trace.hpp>
#include <uni20/core/runtime_precision.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <vector>

namespace uni20::mplapack::detail
{
inline std::size_t auxiliary_size(std::size_t count, std::size_t factor)
{
  if (count > std::numeric_limits<std::size_t>::max() / factor)
    throw std::length_error("MPLAPACK auxiliary workspace size overflow");
  return std::max<std::size_t>(1, count * factor);
}

template <class Adapter, class Scalar>
void orthogonal_factor_impl(bool lq, std::size_t m, std::size_t n, std::span<Scalar> a, std::span<Scalar> q,
                            std::span<Scalar> triangular, Precision p)
{
  auto k = std::min(m, n);
  Adapter::check_size(a.size(), m, n);
  Adapter::check_size(q.size(), lq ? k : m, lq ? n : k);
  Adapter::check_size(triangular.size(), lq ? m : k, lq ? k : n);
  auto pm = Adapter::integer(m), pn = Adapter::integer(n), pk = Adapter::integer(k);
  typename Adapter::precision_scope scope(p);
  if (!k) return;
  using P = typename Adapter::template scalar<Scalar>;
  auto av = Adapter::pack_values(std::span<Scalar const>(a));
  std::vector<P> tau(k);
  Adapter::template workspace<P>(lq ? "gelqf" : "geqrf", [&](P* work, mplapackint length, mplapackint& info) {
    if constexpr (Real<Scalar>)
    {
      if (lq)
        Rgelqf(pm, pn, av.data(), pm, tau.data(), work, length, info);
      else
        Rgeqrf(pm, pn, av.data(), pm, tau.data(), work, length, info);
    }
    else
    {
      if (lq)
        Cgelqf(pm, pn, av.data(), pm, tau.data(), work, length, info);
      else
        Cgeqrf(pm, pn, av.data(), pm, tau.data(), work, length, info);
    }
  });
  auto tr = lq ? m : k, tc = lq ? k : n;
  for (std::size_t j = 0; j < tc; ++j)
    for (std::size_t i = 0; i < tr; ++i)
      triangular[i + j * tr] =
          (lq ? i >= j : i <= j) ? Adapter::get_value(av[i + j * m], p) : scalar_at_precision<Scalar>(0, p);
  Adapter::template workspace<P>(lq ? "orglq" : "orgqr", [&](P* work, mplapackint length, mplapackint& info) {
    if constexpr (Real<Scalar>)
    {
      if (lq)
        Rorglq(pk, pn, pk, av.data(), pm, tau.data(), work, length, info);
      else
        Rorgqr(pm, pk, pk, av.data(), pm, tau.data(), work, length, info);
    }
    else
    {
      if (lq)
        Cunglq(pk, pn, pk, av.data(), pm, tau.data(), work, length, info);
      else
        Cungqr(pm, pk, pk, av.data(), pm, tau.data(), work, length, info);
    }
  });
  auto qr = lq ? k : m, qc = lq ? n : k;
  for (std::size_t j = 0; j < qc; ++j)
    for (std::size_t i = 0; i < qr; ++i)
      q[i + j * qr] = Adapter::get_value(av[i + j * m], p);
  Adapter::unpack_values(av, a, p);
}

template <class Adapter, class Scalar>
void eigh_impl(std::size_t n, std::span<Scalar> a, std::span<make_real_t<Scalar>> values, bool vectors, char triangle,
               Precision p)
{
  Adapter::check_size(a.size(), n, n);
  Adapter::check_size(values.size(), n, 1);
  if (triangle != 'U' && triangle != 'L') throw std::invalid_argument("invalid eigensystem triangle");
  auto pn = Adapter::integer(n);
  typename Adapter::precision_scope scope(p);
  if (!n) return;
  using P = typename Adapter::template scalar<Scalar>;
  auto av = Adapter::pack_values(std::span<Scalar const>(a));
  std::vector<typename Adapter::real> w(n);
  char job = vectors ? 'V' : 'N';
  std::vector<typename Adapter::real> rwork;
  if constexpr (!Real<Scalar>) rwork.resize(auxiliary_size(n, 3));
  Adapter::template workspace<P>("eigh", [&](P* work, mplapackint length, mplapackint& info) {
    if constexpr (Real<Scalar>)
      Rsyev(&job, &triangle, pn, av.data(), pn, w.data(), work, length, info);
    else
      Cheev(&job, &triangle, pn, av.data(), pn, w.data(), work, length, rwork.data(), info);
  });
  Adapter::unpack_values(av, a, p);
  Adapter::unpack_values(w, values, p);
}

template <class Adapter, class Scalar>
void svd_impl(std::size_t m, std::size_t n, std::span<Scalar> a, std::span<make_real_t<Scalar>> values,
              std::span<Scalar> u, std::span<Scalar> vh, char jobu, char jobvt, Precision p)
{
  auto k = std::min(m, n);
  auto valid_job = [](char job) { return job == 'A' || job == 'S' || job == 'N' || job == 'O'; };
  if (!valid_job(jobu) || !valid_job(jobvt) || (jobu == 'O' && jobvt == 'O'))
    throw std::invalid_argument("invalid SVD vector request");
  auto uc = jobu == 'A' ? m : jobu == 'S' ? k : 0;
  auto vr = jobvt == 'A' ? n : jobvt == 'S' ? k : 0;
  Adapter::check_size(a.size(), m, n);
  Adapter::check_size(values.size(), k, 1);
  Adapter::check_size(u.size(), m, uc);
  Adapter::check_size(vh.size(), vr, n);
  auto pm = Adapter::integer(m), pn = Adapter::integer(n);
  auto ldu = std::max<mplapackint>(1, uc ? pm : 1);
  auto ldvt = std::max<mplapackint>(1, Adapter::integer(vr));
  typename Adapter::precision_scope scope(p);
  if (!k)
  {
    for (std::size_t j = 0; j < uc; ++j)
      for (std::size_t i = 0; i < m; ++i)
        u[i + j * m] = scalar_at_precision<Scalar>(i == j ? 1 : 0, p);
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < vr; ++i)
        vh[i + j * vr] = scalar_at_precision<Scalar>(i == j ? 1 : 0, p);
    return;
  }
  using P = typename Adapter::template scalar<Scalar>;
  auto av = Adapter::pack_values(std::span<Scalar const>(a));
  std::vector<P> uv(std::max<std::size_t>(1, u.size())), vv(std::max<std::size_t>(1, vh.size()));
  std::vector<typename Adapter::real> w(k), rwork;
  if constexpr (!Real<Scalar>) rwork.resize(auxiliary_size(k, 5));
  Adapter::template workspace<P>("gesvd", [&](P* work, mplapackint length, mplapackint& info) {
    if constexpr (Real<Scalar>)
      Rgesvd(&jobu, &jobvt, pm, pn, av.data(), pm, w.data(), uv.data(), ldu, vv.data(), ldvt, work, length, info);
    else
      Cgesvd(&jobu, &jobvt, pm, pn, av.data(), pm, w.data(), uv.data(), ldu, vv.data(), ldvt, work, length,
             rwork.data(), info);
  });
  Adapter::unpack_values(av, a, p);
  Adapter::unpack_values(w, values, p);
  uv.resize(u.size());
  vv.resize(vh.size());
  Adapter::unpack_values(uv, u, p);
  Adapter::unpack_values(vv, vh, p);
}

template <class Adapter, class Scalar>
void eigen_impl(std::size_t n, std::span<Scalar> a, std::span<complex<make_real_t<Scalar>>> values,
                std::span<complex<make_real_t<Scalar>>> vectors, bool compute_vectors, Precision p)
{
  Adapter::check_size(a.size(), n, n);
  Adapter::check_size(values.size(), n, 1);
  Adapter::check_size(vectors.size(), compute_vectors ? n : 0, n);
  auto pn = Adapter::integer(n);
  typename Adapter::precision_scope scope(p);
  if (!n) return;
  using P = typename Adapter::template scalar<Scalar>;
  auto av = Adapter::pack_values(std::span<Scalar const>(a));
  std::vector<P> vl(1), vr(std::max<std::size_t>(1, vectors.size()));
  char job = compute_vectors ? 'V' : 'N';
  auto ldvr = compute_vectors ? pn : mplapackint{1};
  if constexpr (Real<Scalar>)
  {
    std::vector<P> wr(n), wi(n);
    Adapter::template workspace<P>("geev", [&](P* work, mplapackint length, mplapackint& info) {
      Rgeev("N", &job, pn, av.data(), pn, wr.data(), wi.data(), vl.data(), 1, vr.data(), ldvr, work, length, info);
    });
    for (std::size_t j = 0; j < n; ++j)
      values[j] = complex<make_real_t<Scalar>>(Adapter::get_value(wr[j], p), Adapter::get_value(wi[j], p));
    if (compute_vectors)
      for (std::size_t j = 0; j < n; ++j)
      {
        if (wi[j] == 0)
          for (std::size_t i = 0; i < n; ++i)
            vectors[i + j * n] = complex<make_real_t<Scalar>>(Adapter::get_value(vr[i + j * n], p),
                                                              scalar_at_precision<make_real_t<Scalar>>(0, p));
        else
        {
          CHECK(wi[j] > 0 && j + 1 < n, "invalid conjugate eigenvector pair");
          for (std::size_t i = 0; i < n; ++i)
          {
            auto v = complex<make_real_t<Scalar>>(Adapter::get_value(vr[i + j * n], p),
                                                  Adapter::get_value(vr[i + (j + 1) * n], p));
            vectors[i + j * n] = v;
            vectors[i + (j + 1) * n] = uni20::conj(v);
          }
          ++j;
        }
      }
  }
  else
  {
    std::vector<P> w(n);
    std::vector<typename Adapter::real> rwork(auxiliary_size(n, 2));
    Adapter::template workspace<P>("geev", [&](P* work, mplapackint length, mplapackint& info) {
      Cgeev("N", &job, pn, av.data(), pn, w.data(), vl.data(), 1, vr.data(), ldvr, work, length, rwork.data(), info);
    });
    Adapter::unpack_values(w, values, p);
    if (compute_vectors) Adapter::unpack_values(vr, vectors, p);
  }
  Adapter::unpack_values(av, a, p);
}

} // namespace uni20::mplapack::detail
