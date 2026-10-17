#pragma once

/**
 * \file binary80_common.hpp
 * \brief Binary80 scalar eligibility and logical host packing for BLAS/LAPACK adapters.
 */
#include <uni20/backend/mplapack/binary80.hpp>
#include <uni20/linalg/backends/packed_matrix.hpp>

namespace uni20::linalg
{
namespace mplapack_binary80_detail
{
template <class T>
concept Binary80Scalar = std::same_as<T, float80> || std::same_as<T, complex160>;

// GEMM needs no complex division. LU and decompositions require the validated native provider
// specialization; otherwise ordinary dispatch can select the CPU backend.
template <class T>
concept Binary80LuScalar =
    std::same_as<T, float80> || (UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX != 0 && std::same_as<T, complex160>);

template <class Span> auto pack(Span const& span, bool read = true)
{
  return packed_detail::pack(span, native_precision<typename Span::value_type>(), read);
}
using packed_detail::unpack;
using packed_detail::values;
} // namespace mplapack_binary80_detail
} // namespace uni20::linalg
