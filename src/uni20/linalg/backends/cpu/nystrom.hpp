#pragma once

#include <uni20/linalg/detail/nystrom_scaling.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>

namespace uni20::linalg
{
template <class M, class V, class R>
concept HostNystromFactors =
    MutableRankedMdspecLike<M, 2> && RankedMdspecLike<V, 1> && HostWritableMdspec<M> && HostReadableMdspec<V> &&
    std::same_as<typename M::value_type, R> && std::same_as<typename V::value_type, R> &&
    (std::same_as<R, float> || std::same_as<R, double>);

template <class M, class V, class R>
  requires HostNystromFactors<M, V, R>
consteval auto kernel_accepts_types(CpuReferenceBackend, nystrom_whiten_op<R>, M&, V&)
{
  return kernel_types_maybe;
}

template <class M, class V, class R>
  requires HostNystromFactors<M, V, R>
KernelAttempt try_kernel(CpuReferenceBackend, nystrom_whiten_op<R> op, M& columns, V& values)
{
  auto write = acquire_host_write_access_sync(columns);
  auto read = acquire_host_read_access_sync(values);
  auto a = write.mdspan();
  auto d = read.mdspan();
  auto const width = d.extent(0);
  for (index_type j = 0; j < width; ++j)
  {
    R const scale = detail::randomized::whitening_scale(d[j], d[width - 1], op.tolerance);
    for (index_type i = 0; i < width; ++i)
      a[i, j] *= scale;
  }
  return KernelAttempt::success;
}
} // namespace uni20::linalg
