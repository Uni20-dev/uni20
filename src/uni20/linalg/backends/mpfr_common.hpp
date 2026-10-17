#pragma once
#include <uni20/core/mpcomplex.hpp>
#include <uni20/linalg/backends/packed_matrix.hpp>

namespace uni20::linalg::mpfr_detail
{
template <class T>
concept RuntimeScalar = std::same_as<T, mpreal> || std::same_as<T, mpcomplex>;
using packed_detail::pack;
using packed_detail::unpack;
using packed_detail::values;
}
