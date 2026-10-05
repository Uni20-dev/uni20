#pragma once

/// \file gtest.hpp
/// \brief GoogleTest integration for ULP-based floating-point comparisons.
///
/// This header defines two macros, `EXPECT_FLOATING_EQ` and `ASSERT_FLOATING_EQ`,
/// for use inside GoogleTest unit tests. They extend the standard GTest
/// floating-point comparison macros (`EXPECT_FLOAT_EQ`, `EXPECT_DOUBLE_EQ`) to:
///
/// - Work with IEEE binary32, binary64, configured native fp80, binary128, and optional mpreal.
/// - Work with `uni20::complex<T>` over the native real scalar types.
/// - Allow explicit specification of ULP tolerance.
/// - Default to a tolerance of 4 ULPs if none is provided, matching GoogleTest.
///
/// \details
/// Usage patterns:
/// \code
/// float a = 1.0f;
/// float b = std::nextafter(a, 2.0f);
///
/// // Default tolerance of 4 ULPs
/// EXPECT_FLOATING_EQ(a, b);
///
/// // Explicit tolerance of 1 ULP
/// EXPECT_FLOATING_EQ(a, b, 1);
///
/// // ASSERT_ variant aborts the current test case on failure
/// ASSERT_FLOATING_EQ(a, b, 1);
/// \endcode
///
/// Failure output shows:
/// - The source file and line number
/// - The compared expressions and their evaluated values
/// - The allowed tolerance in ULPs
/// - The actual ULP distance, computed via `uni20::check::float_distance`
/// - For mpreal, operand states/precisions and the reason an invalid pair cannot be compared
///
/// \note These macros are intended for unit tests only.
/// For assertions in library code, use `CHECK_FLOATING_EQ` / `PRECONDITION_FLOATING_EQ`
/// from `trace.hpp`.
///
/// \pre The compared type must satisfy `uni20::check::UlpComparable`.
/// \post On failure, the macros report via GoogleTest (`ADD_FAILURE` or `FAIL`).
///
/// \see CHECK_FLOATING_EQ, PRECONDITION_FLOATING_EQ, uni20::check::float_distance

#include "floating_eq.hpp"
#include "trace.hpp"
#include <gtest/gtest.h>
#include <uni20/core/scalar_io.hpp>

namespace uni20::check::detail
{
template <class T> std::string floating_operand_text(T const& value) { return uni20::format_scalar(value); }

#if UNI20_ENABLE_MPFR
inline std::string floating_operand_text(mpreal const& value)
{
  if (!value.initialized()) return "<unset>";
  auto text = value.to_string();
  return value.is_exact() ? text + " [exact]" : text + " [" + std::to_string(value.precision().bit_count()) + " bits]";
}
#endif

template <UlpComparable T>
::testing::AssertionResult floating_eq_assertion(char const* name, char const* left, char const* right, T const& a,
                                                 T const& b, std::int64_t ulps)
{
  if (ulps < 0) return ::testing::AssertionFailure() << name << " requires non-negative ULP tolerance, got " << ulps;
  if (FloatingULP<T>::eq(a, b, ulps)) return ::testing::AssertionSuccess();

  auto result = ::testing::AssertionFailure();
  result << name << " failed\n  " << left << " = " << floating_operand_text(a) << "\n  " << right << " = "
         << floating_operand_text(b) << "\n  allowed tolerance: " << ulps << " ULP\n  actual distance: ";
  auto const distance = float_abs_distance(a, b);
  if (distance == std::numeric_limits<long long>::max())
    result << "unrepresentable or exceeds diagnostic range";
  else
    result << distance;
#if UNI20_ENABLE_MPFR
  if constexpr (std::same_as<T, mpreal>)
    if (auto reason = compare_mpreal(a, b, ulps).reason) result << "\n  reason: " << reason;
#endif
  return result;
}
} // namespace uni20::check::detail

#define EXPECT_FLOATING_EQ(a, b, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    auto va = (a);                                                                                                     \
    auto vb = (b);                                                                                                     \
    auto ulps = ::trace::detail::get_ulps(va, vb __VA_OPT__(, __VA_ARGS__));                                           \
    auto result = ::uni20::check::detail::floating_eq_assertion("EXPECT_FLOATING_EQ", #a, #b, va, vb, ulps);           \
    if (!result)                                                                                                       \
    {                                                                                                                  \
      ADD_FAILURE() << result.message();                                                                               \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define ASSERT_FLOATING_EQ(a, b, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    auto va = (a);                                                                                                     \
    auto vb = (b);                                                                                                     \
    auto ulps = ::trace::detail::get_ulps(va, vb __VA_OPT__(, __VA_ARGS__));                                           \
    auto result = ::uni20::check::detail::floating_eq_assertion("ASSERT_FLOATING_EQ", #a, #b, va, vb, ulps);           \
    if (!result)                                                                                                       \
    {                                                                                                                  \
      FAIL() << result.message();                                                                                      \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)
