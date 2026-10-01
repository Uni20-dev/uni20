#pragma once

/**
 * \file operation_scalar.hpp
 * \ingroup linalg
 * \brief Numeric coefficient sources shared by synchronous and asynchronous operations.
 */

#include <uni20/core/runtime_precision.hpp>
#if UNI20_ENABLE_MPFR
#include <uni20/core/exact_constant.hpp>
#endif

#include <concepts>

namespace uni20::linalg
{
/// \brief Numeric argument that an operation can convert to its scalar type.
/// \details Runtime-precision scalars also accept explicit construction from
///          exact numeric sources, but not from precision or storage tags.
template <class Value, class Scalar>
concept OperationScalar =
    std::convertible_to<Value, Scalar>
#if UNI20_ENABLE_MPFR
    || (has_runtime_precision_v<Scalar> && ExactRationalSource<Value> && std::constructible_from<Scalar, Value>)
#endif
    ;
} // namespace uni20::linalg
