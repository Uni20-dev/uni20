#include "../numerics/precision_cases.hpp"
#include <mplapack_config.h>
#include <uni20/common/gtest.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/backends/cpu/matrix_exponential.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/lq.hpp>
#include <uni20/linalg/ops/matrix_norm.hpp>
#include <uni20/linalg/ops/qr.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>
#include <uni20/tensor/reductions.hpp>
#include <uni20/tensor/tensor.hpp>
#include <uni20/tensor/transform.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>

#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_LDBL)

TEST(MplapackBinary128CpuOpsTest, SkipsLongDoubleAliasMode)
{
  GTEST_SKIP() << "configured MPLAPACK binary128 mode aliases long double";
}

#else

namespace
{

using Binary128 = mplapack_binary128_t;
using matrix_type = uni20::DenseMatrix<Binary128>;

Binary128 abs_error(Binary128 actual, Binary128 expected) { return std::abs(actual - expected); }

Binary128 tolerance() { return static_cast<Binary128>(1.0e-25L); }

Binary128 binary_power_of_two(int exponent)
{
  return uni20::test::PrecisionContext<Binary128, false>::power_of_two(exponent);
}

Binary128 below_double_resolution_gap() { return binary_power_of_two(-80); }

void expect_gap_is_binary128_only(Binary128 gap)
{
  EXPECT_TRUE(Binary128{1} + gap > Binary128{1});
  EXPECT_EQ(static_cast<double>(Binary128{1} + gap), 1.0);
}

} // namespace

TEST(MplapackBinary128CpuOpsTest, MatrixExponentialPrescalesWithinBinary128)
{
  Binary128 const large = binary_power_of_two(uni20::numeric_limits<Binary128>::max_exponent - 2);
  ASSERT_TRUE(uni20::isfinite(large));

  matrix_type matrix(2, 2);
  uni20::fill(matrix, Binary128{});
  matrix[0, 1] = large;

  auto const result = uni20::linalg::backends::cpu::matrix_exponential(matrix, Binary128{1});

  EXPECT_FLOATING_EQ((result[0, 0]), Binary128{1});
  EXPECT_FLOATING_EQ((result[1, 0]), Binary128{});
  EXPECT_FLOATING_EQ((result[1, 1]), Binary128{1});
  EXPECT_TRUE(abs_error((result[0, 1] - large) / large, Binary128{}) <= tolerance());
}

TEST(MplapackBinary128CpuOpsTest, ExactSvdPreservesRealAndComplexBinary128Values)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::DenseMatrix<Binary128> real_matrix(2, 2);
  uni20::fill(real_matrix, Binary128{});
  real_matrix[0, 0] = Binary128{2} + delta;
  real_matrix[1, 1] = Binary128{1};
  auto real_values = uni20::linalg::singular_values(real_matrix);
  auto real_left = uni20::linalg::svd_left(real_matrix);
  auto real_result = uni20::linalg::svd(real_matrix);

  EXPECT_FLOATING_EQ(real_values[0], Binary128{2} + delta);
  EXPECT_FLOATING_EQ(real_values[1], Binary128{1});
  EXPECT_FLOATING_EQ(real_left.singular_values[0], Binary128{2} + delta);
  EXPECT_FLOATING_EQ(real_left.singular_values[1], Binary128{1});
  EXPECT_FLOATING_EQ(real_result.singular_values[0], Binary128{2} + delta);
  EXPECT_FLOATING_EQ(real_result.singular_values[1], Binary128{1});

  using Complex = uni20::complex<Binary128>;
  uni20::DenseMatrix<Complex> complex_matrix(2, 2);
  uni20::fill(complex_matrix, Complex{});
  complex_matrix[0, 0] = Complex{Binary128{}, Binary128{2} + delta};
  complex_matrix[1, 1] = Complex{Binary128{1}, Binary128{}};
  auto complex_right = uni20::linalg::svd_right(complex_matrix);
  auto complex_result = uni20::linalg::svd(complex_matrix);

  EXPECT_FLOATING_EQ(complex_right.singular_values[0], Binary128{2} + delta);
  EXPECT_FLOATING_EQ(complex_right.singular_values[1], Binary128{1});
  EXPECT_FLOATING_EQ(complex_result.singular_values[0], Binary128{2} + delta);
  EXPECT_FLOATING_EQ(complex_result.singular_values[1], Binary128{1});
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      Complex reconstructed{};
      for (uni20::index_type inner = 0; inner < 2; ++inner)
      {
        reconstructed += complex_result.left_singular_vectors[row, inner] * complex_result.singular_values[inner] *
                         complex_result.right_singular_vectors_adjoint[inner, column];
      }
      EXPECT_TRUE(std::abs(reconstructed - complex_matrix[row, column]) <= tolerance());
    }
  }
}

TEST(MplapackBinary128CpuOpsTest, ReducedQrAndLqPreserveBinary128Values)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  matrix_type matrix(2, 2);
  uni20::fill(matrix, Binary128{});
  matrix[0, 0] = Binary128{2} + delta;
  matrix[1, 1] = Binary128{1};

  auto qr_result = uni20::linalg::qr(matrix);
  auto lq_result = uni20::linalg::lq(matrix);

  Binary128 const qr_reconstructed = qr_result.q[0, 0] * qr_result.r[0, 0] + qr_result.q[0, 1] * qr_result.r[1, 0];
  Binary128 const lq_reconstructed = lq_result.l[0, 0] * lq_result.q[0, 0] + lq_result.l[0, 1] * lq_result.q[1, 0];
  EXPECT_FLOATING_EQ(qr_reconstructed, Binary128{2} + delta);
  EXPECT_FLOATING_EQ(lq_reconstructed, Binary128{2} + delta);
  EXPECT_EQ(static_cast<double>(qr_reconstructed), 2.0);
  EXPECT_EQ(static_cast<double>(lq_reconstructed), 2.0);
}

TEST(MplapackBinary128CpuOpsTest, TruncatedSvdUsesBinary128PolicyAndStatistics)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::DenseMatrix<Binary128> matrix(2, 2);
  uni20::fill(matrix, Binary128{});
  matrix[0, 0] = Binary128{2} + delta;
  matrix[1, 1] = Binary128{1};
  auto result =
      uni20::linalg::truncated_svd(matrix, uni20::linalg::SvdTruncationPolicy<Binary128>{.maximum_retained_extent = 1});

  static_assert(std::same_as<decltype(result.truncation.original_squared_norm), Binary128>);
  EXPECT_EQ(result.truncation.available_rank, 2);
  EXPECT_EQ(result.truncation.retained_rank, 1);
  EXPECT_FLOATING_EQ(result.singular_values[0], Binary128{2} + delta);
  EXPECT_TRUE(abs_error(result.truncation.discarded_weight,
                        Binary128{1} / ((Binary128{2} + delta) * (Binary128{2} + delta) + Binary128{1})) <=
              tolerance());
}

#endif
