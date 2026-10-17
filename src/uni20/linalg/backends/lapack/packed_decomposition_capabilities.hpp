#pragma once

/**
 * \file packed_decomposition_capabilities.hpp
 * \brief Per-operation eligibility for compiled value-converting provider adapters.
 */
#include <span>
#include <uni20/config.hpp>
#include <uni20/core/runtime_precision.hpp>
#include <uni20/core/scalar_concepts.hpp>
#if UNI20_HAS_MPLAPACK_MPREAL
#include <uni20/backend/mplapack/mpfr.hpp>
#endif
#if UNI20_HAS_MPLAPACK_FLOAT80
#include <uni20/backend/mplapack/binary80.hpp>
#endif

namespace uni20::linalg::packed_detail
{
// Fixed scalars omit precision at the kernel boundary; runtime scalars require it.
template <class S, class... P>
concept KernelPrecision = (has_runtime_precision_v<S> && sizeof...(P) == 1 && (std::same_as<P, Precision> && ...)) ||
                          (!has_runtime_precision_v<S> && sizeof...(P) == 0);
template <class S> Precision working_precision() { return native_precision<S>(); }
template <class S> Precision working_precision(Precision p)
{
  require_precision<S>(p);
  return p;
}
template <class S>
concept QrScalar = requires(std::size_t n, std::span<S> a, Precision p) { mplapack::qr(n, n, a, a, a, p); };
template <class S>
concept LqScalar = requires(std::size_t n, std::span<S> a, Precision p) { mplapack::lq(n, n, a, a, a, p); };
template <class S>
concept EighScalar = requires(std::size_t n, std::span<S> a, std::span<make_real_t<S>> v, Precision p) {
  mplapack::eigh(n, a, v, true, 'U', p);
};
template <class S>
concept SvdScalar = requires(std::size_t n, std::span<S> a, std::span<make_real_t<S>> v, Precision p) {
  mplapack::svd(n, n, a, v, a, a, 'S', 'S', p);
};
template <class S>
concept EigenScalar = requires(std::size_t n, std::span<S> a, std::span<complex<make_real_t<S>>> z, Precision p) {
  mplapack::eigen(n, a, z, z, true, p);
};
} // namespace uni20::linalg::packed_detail
