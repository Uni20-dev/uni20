#pragma once

/**
 * \file common.hpp
 * \ingroup linalg
 * \brief Shared helpers for LAPACK operation-tag backend adapters.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/kernel_attempt.hpp>
#include <uni20/tensor/access.hpp>

#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg::lapack_detail
{

/// \brief Invoke a LAPACK attempt with host-writable mdspans under simultaneous leases.
template <class Function, uni20::HostWritableMdspec... Mdspecs>
  requires std::same_as<std::invoke_result_t<Function&&, uni20::host_write_mdspan_t<Mdspecs>&...>, KernelAttempt>
KernelAttempt with_host_write_mdspans(Function&& function, Mdspecs&... mdspecs)
{
  auto accesses = std::tuple{acquire_host_write_access_sync(mdspecs)...};
  return std::apply(
      [&](auto&... access) -> KernelAttempt {
        auto mdspans = std::tuple{access.mdspan()...};
        return std::apply(
            [&](auto&... mdspan) -> KernelAttempt { return std::invoke(std::forward<Function>(function), mdspan...); },
            mdspans);
      },
      accesses);
}

/// \brief Convert a provider workspace query result to the configured LAPACK integer type.
template <class Scalar> blas_int workspace_size(Scalar const& query)
{
  auto const recommended = uni20::real(query);
  using real_type = std::remove_cv_t<decltype(recommended)>;
  CHECK(uni20::isfinite(recommended), recommended);
  CHECK(recommended >= real_type{1}, recommended);

  auto const maximum = static_cast<real_type>(uni20::numeric_limits<blas_int>::max());
  auto const precision = uni20::numeric_limits<real_type>::digits(recommended);
  blas_int result;
  if (precision < uni20::numeric_limits<blas_int>::digits())
  {
    // A low-precision real may round max(blas_int) up to the first
    // unrepresentable integer, so equality is not a safe conversion bound.
    CHECK(recommended < maximum, recommended);
    result = static_cast<blas_int>(recommended);

    // Some providers round an integer workspace down when storing it in WORK(1).
    // Once the real type loses low integer bits, allocate through the last
    // integer below its next representable value.
    auto const missing_bits = static_cast<int>(std::bit_width(static_cast<std::make_unsigned_t<blas_int>>(result))) -
                              precision;
    if (missing_bits > 0) result += (blas_int{1} << missing_bits) - 1;
  }
  else
  {
    CHECK(recommended <= maximum, recommended);
    result = static_cast<blas_int>(recommended);
  }

  CHECK(result >= 1, result);
  CHECK(std::in_range<std::size_t>(result), result);
  return result;
}

/// \brief Multiply allocation extents without wrapping `std::size_t`.
constexpr std::optional<std::size_t> try_size_product(std::size_t left, std::size_t right) noexcept
{
  if (left != 0 && right > uni20::numeric_limits<std::size_t>::max() / left)
  {
    return std::nullopt;
  }
  return left * right;
}

/// \brief Evaluate a workspace term in the range shared by the host and LAPACK.
constexpr std::optional<blas_int> try_lapack_workspace_size(std::size_t count, std::size_t multiplier,
                                                            std::size_t extra = 0) noexcept
{
  auto const product = try_size_product(count, multiplier);
  if (!product || extra > uni20::numeric_limits<std::size_t>::max() - *product) return std::nullopt;
  auto const result = uni20::blas::try_blas_int(*product + extra);
  if (!uni20::blas::is_valid_blas_int(result)) return std::nullopt;
  return result;
}

} // namespace uni20::linalg::lapack_detail
