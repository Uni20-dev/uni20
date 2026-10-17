#pragma once

#include "decompositions_impl.hpp"

namespace uni20::mplapack::detail
{
template <class Adapter, class S>
std::size_t gelss_impl(std::size_t m, std::size_t n, std::size_t nrhs, std::span<S> a, std::span<S> b,
                       std::span<make_real_t<S>> values, make_real_t<S> const& cutoff, std::size_t& rank, Precision p)
{
  auto k = std::min(m, n), rows_b = std::max(m, n);
  Adapter::check_size(a.size(), m, n);
  Adapter::check_size(b.size(), rows_b, nrhs);
  Adapter::check_size(values.size(), k, 1);
  if (!k) throw std::invalid_argument("packed GELSS requires nonzero dimensions");
  auto pm = Adapter::integer(m), pn = Adapter::integer(n), prhs = Adapter::integer(nrhs);
  auto ldb = Adapter::integer(rows_b);
  typename Adapter::precision_scope scope(p);
  using P = typename Adapter::template scalar<S>;
  auto av = Adapter::pack_values(std::span<S const>(a));
  auto bv = Adapter::pack_values(std::span<S const>(b));
  // Providers still need a valid address when there are no RHS columns.
  if (bv.empty()) bv.resize(1);
  auto threshold = Adapter::pack_values(std::span(&cutoff, 1));
  std::vector<typename Adapter::real> singular(k), rwork;
  if constexpr (Complex<S>) rwork.resize(auxiliary_size(k, 5));
  mplapackint provider_rank = 0;
  auto info = Adapter::template workspace_info<P>([&](P* work, mplapackint length, mplapackint& code) {
    if constexpr (Real<S>)
      Rgelss(pm, pn, prhs, av.data(), pm, bv.data(), ldb, singular.data(), threshold[0], provider_rank, work, length,
             code);
    else
      Cgelss(pm, pn, prhs, av.data(), pm, bv.data(), ldb, singular.data(), threshold[0], provider_rank, work, length,
             rwork.data(), code);
  });
  if (info) return static_cast<std::size_t>(info);
  rank = static_cast<std::size_t>(provider_rank);
  Adapter::unpack_values(av, a, p);
  bv.resize(b.size());
  Adapter::unpack_values(bv, b, p);
  Adapter::unpack_values(singular, values, p);
  return 0;
}

template <class Adapter, class S>
make_real_t<S> gecon_impl(std::size_t n, std::span<S const> a, make_real_t<S> const& anorm, char norm, Precision p)
{
  Adapter::check_size(a.size(), n, n);
  if (norm != '1' && norm != 'I') throw std::invalid_argument("condition norm must be one or infinity");
  auto pn = Adapter::integer(n);
  typename Adapter::precision_scope scope(p);
  if (!n) return scalar_at_precision<make_real_t<S>>(1, p);
  auto av = Adapter::pack_values(a);
  auto norm_value = Adapter::pack_values(std::span(&anorm, 1));
  using P = typename Adapter::template scalar<S>;
  typename Adapter::real rcond;
  mplapackint info = 0;
  if constexpr (Real<S>)
  {
    std::vector<P> work(auxiliary_size(n, 4));
    std::vector<mplapackint> iwork(n);
    Rgecon(&norm, pn, av.data(), pn, norm_value[0], rcond, work.data(), iwork.data(), info);
  }
  else
  {
    std::vector<P> work(auxiliary_size(n, 2));
    std::vector<typename Adapter::real> rwork(auxiliary_size(n, 2));
    Cgecon(&norm, pn, av.data(), pn, norm_value[0], rcond, work.data(), rwork.data(), info);
  }
  if (info < 0) throw std::logic_error("MPLAPACK rejected validated GECON arguments");
  return Adapter::get_value(rcond, p);
}
} // namespace uni20::mplapack::detail
