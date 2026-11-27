#include <uni20/linalg/ops/cuda_jacobi.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace
{

bool has_cuda_device()
{
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

template <class Scalar> class CusolverJacobiTest : public testing::Test {};
using RealScalars = testing::Types<float, double>;
TYPED_TEST_SUITE(CusolverJacobiTest, RealScalars);

TYPED_TEST(CusolverJacobiTest, EighPreservesDeviceInputAndUsesChosenTriangle)
{
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  auto& resources = runtime.device_resources(0);

  using Scalar = TypeParam;
  constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
  uni20::DenseMatrix<Scalar> input(3, 3);
  input[0, 0] = 3;
  input[1, 1] = 3;
  input[2, 2] = -1;
  input[1, 0] = 1;
  input[0, 1] = 99;

  auto device_input = uni20::to_device(input, resources);
  auto [factors, convergence] = uni20::linalg::eigh_jacobi(device_input, {.tolerance = epsilon, .maximum_sweeps = 15},
                                                           uni20::linalg::MatrixTriangle::Lower);
  ASSERT_TRUE(convergence.converged);
  EXPECT_GE(convergence.residual, 0);
  EXPECT_LE(convergence.residual, 8 * epsilon);
  EXPECT_GT(convergence.sweeps, 0);
  EXPECT_LE(convergence.sweeps, 15);

  auto values = uni20::to_host(factors.eigenvalues);
  auto vectors = uni20::to_host(factors.eigenvectors);
  auto preserved = uni20::to_host(device_input);
  EXPECT_NEAR(values[0], -1, 16 * epsilon);
  EXPECT_NEAR(values[1], 2, 16 * epsilon);
  EXPECT_NEAR(values[2], 4, 16 * epsilon);
  for (uni20::index_type row = 0; row < 3; ++row)
    for (uni20::index_type col = 0; col < 3; ++col)
    {
      EXPECT_EQ((preserved[row, col]), (input[row, col]));
      Scalar product = 0;
      for (uni20::index_type k = 0; k < 3; ++k)
        product += (row >= k ? input[row, k] : input[k, row]) * vectors[k, col];
      EXPECT_NEAR(product, (vectors[row, col] * values[col]), 16 * epsilon);
    }
}

TYPED_TEST(CusolverJacobiTest, ReducedSvdReconstructsTallAndWideMatrices)
{
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  auto& resources = runtime.device_resources(0);

  using Scalar = TypeParam;
  constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
  for (bool wide : {false, true})
  {
    uni20::index_type const rows = wide ? 2 : 3;
    uni20::index_type const cols = wide ? 3 : 2;
    uni20::DenseMatrix<Scalar> input(rows, cols);
    for (uni20::index_type row = 0; row < rows; ++row)
      for (uni20::index_type col = 0; col < cols; ++col)
        input[row, col] = static_cast<Scalar>((row + 1) * (col + 2) + (row == col ? 1 : 0));

    auto device_input = uni20::to_device(input, resources);
    auto [factors, convergence] = uni20::linalg::svd_jacobi(device_input);
    ASSERT_TRUE(convergence.converged);
    EXPECT_TRUE(std::isfinite(convergence.residual));
    EXPECT_GT(convergence.sweeps, 0);

    auto u = uni20::to_host(factors.left_singular_vectors);
    auto s = uni20::to_host(factors.singular_values);
    auto vh = uni20::to_host(factors.right_singular_vectors_adjoint);
    auto preserved = uni20::to_host(device_input);
    ASSERT_EQ(u.extent(0), rows);
    ASSERT_EQ(u.extent(1), 2);
    ASSERT_EQ(s.extent(0), 2);
    ASSERT_EQ(vh.extent(0), 2);
    ASSERT_EQ(vh.extent(1), cols);
    EXPECT_GT(s[0], s[1]);
    EXPECT_GT(s[1], 0);
    for (uni20::index_type row = 0; row < rows; ++row)
      for (uni20::index_type col = 0; col < cols; ++col)
      {
        EXPECT_EQ((preserved[row, col]), (input[row, col]));
        Scalar product = 0;
        for (uni20::index_type k = 0; k < 2; ++k)
          product += u[row, k] * s[k] * vh[k, col];
        EXPECT_NEAR(product, (input[row, col]), 64 * epsilon);
      }
  }
}

TEST(CusolverJacobi, ReportsSweepLimitForBothDrivers)
{
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  auto& resources = runtime.device_resources(0);

  constexpr uni20::index_type n = 37;
  uni20::DenseMatrix<double> input(n, n);
  for (uni20::index_type row = 0; row < n; ++row)
    for (uni20::index_type col = 0; col < n; ++col)
      input[row, col] = std::cos(static_cast<double>((row + 1) * (col + 1))) + (row == col ? 3.0 : 0.0);
  auto matrix = uni20::to_device(input, resources);
  uni20::linalg::CudaJacobiOptions const options{.tolerance = 1e-14, .maximum_sweeps = 1};

  auto eig = uni20::linalg::eigh_jacobi(matrix, options);
  auto svd = uni20::linalg::svd_jacobi(matrix, options);
  for (auto const& status : {eig.convergence, svd.convergence})
  {
    EXPECT_FALSE(status.converged);
    EXPECT_EQ(status.sweeps, 1);
    EXPECT_GT(status.residual, 1e-6);
    EXPECT_TRUE(std::isfinite(status.residual));
  }
}

TEST(CusolverJacobi, ValidatesOptionsAndPreservesEmptyShapes)
{
  if (!has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  auto& resources = runtime.device_resources(0);

  uni20::CudaMatrix<double> square(uni20::uninitialized, resources, 0, 0);
  EXPECT_THROW((void)uni20::linalg::eigh_jacobi(square, {.maximum_sweeps = 0}), std::invalid_argument);
  EXPECT_THROW((void)uni20::linalg::svd_jacobi(square, {.tolerance = std::numeric_limits<double>::infinity()}),
               std::invalid_argument);
  auto eig = uni20::linalg::eigh_jacobi(square);
  EXPECT_TRUE(eig.convergence.converged);
  EXPECT_EQ(eig.convergence.sweeps, 0);
  EXPECT_EQ(eig.factors.eigenvalues.extent(0), 0);

  uni20::CudaMatrix<double> rectangular(uni20::uninitialized, resources, 0, 4);
  EXPECT_THROW((void)uni20::linalg::eigh_jacobi(rectangular), std::invalid_argument);
  auto svd = uni20::linalg::svd_jacobi(rectangular);
  EXPECT_TRUE(svd.convergence.converged);
  EXPECT_EQ(svd.convergence.sweeps, 0);
  EXPECT_EQ(svd.factors.singular_values.extent(0), 0);
  EXPECT_EQ(svd.factors.right_singular_vectors_adjoint.extent(0), 0);
  EXPECT_EQ(svd.factors.right_singular_vectors_adjoint.extent(1), 4);
}

} // namespace
