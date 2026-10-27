#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

namespace
{
#include "readme_product_step.inc"

TEST(ReadmeExample, ProductStepWithOrdinaryAndAsyncMatrices)
{
  using matrix_type = uni20::DenseMatrix<double>;
  matrix_type lhs(2, 3);
  matrix_type rhs(3, 2);
  for (uni20::index_type row = 0; row < lhs.rows(); ++row)
    for (uni20::index_type col = 0; col < lhs.cols(); ++col)
      lhs[row, col] = 1 + 3 * row + col;
  for (uni20::index_type row = 0; row < rhs.rows(); ++row)
    for (uni20::index_type col = 0; col < rhs.cols(); ++col)
      rhs[row, col] = 7 + 2 * row + col;
  matrix_type output;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<matrix_type> async_lhs = lhs;
  uni20::async::Async<matrix_type> async_rhs = rhs;
  uni20::async::Async<matrix_type> async_output;

#include "readme_product_step_calls.inc"

  auto const& async_result = async_output.get_wait(scheduler);
  // Independent exact fixture: 1.5 * [[58, 64], [139, 154]].
  double const expected[2][2] = {{87, 96}, {208.5, 231}};
  matrix_type const* const results[] = {&output, &async_result};
  for (auto const* result : results)
  {
    ASSERT_EQ(result->rows(), 2);
    ASSERT_EQ(result->cols(), 2);
    for (uni20::index_type row = 0; row < 2; ++row)
      for (uni20::index_type col = 0; col < 2; ++col)
        EXPECT_EQ(((*result)[row, col]), expected[row][col]);
  }
}
} // namespace
