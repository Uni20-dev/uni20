#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <type_traits>

namespace
{

template <class Scalar> class CusolverEighTest : public testing::Test {};

using RealScalars = testing::Types<float, double>;
TYPED_TEST_SUITE(CusolverEighTest, RealScalars);

TYPED_TEST(CusolverEighTest, DeviceEigensystemReadsOnlyTheSelectedTriangle)
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "CUDA device unavailable";

  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  using Scalar = TypeParam;
  using uni20::linalg::MatrixTriangle;
  constexpr std::array<Scalar, 3> expected{0, 2, 4};
  constexpr Scalar tolerance = 32 * std::numeric_limits<Scalar>::epsilon();

  uni20::Tensor<Scalar, 2> matrix(3, 3);
  matrix[0, 0] = 3;
  matrix[1, 1] = 3;
  matrix[1, 0] = 1;
  matrix[0, 1] = 1;
  for (auto triangle : {MatrixTriangle::Upper, MatrixTriangle::Lower})
  {
    auto input = matrix;
    if (triangle == MatrixTriangle::Upper)
      input[2, 0] = 17;
    else
      input[0, 2] = 17;

    auto device_input = uni20::to_device(input, resources);
    auto result = uni20::linalg::eigh(device_input, triangle);
    static_assert(std::is_same_v<decltype(result.eigenvalues), uni20::CudaTensor<Scalar, 1>>);
    static_assert(std::is_same_v<decltype(result.eigenvectors), uni20::CudaMatrix<Scalar, uni20::ColumnMajor>>);
    auto values = uni20::to_host(result.eigenvalues);
    auto vectors = uni20::to_host(result.eigenvectors);
    auto preserved = uni20::to_host(device_input);

    for (uni20::index_type col = 0; col < 3; ++col)
    {
      EXPECT_NEAR(values[col], expected[static_cast<std::size_t>(col)], tolerance);
      for (uni20::index_type row = 0; row < 3; ++row)
      {
        EXPECT_EQ((preserved[row, col]), (input[row, col]));
        Scalar av = 0;
        Scalar norm = 0;
        for (uni20::index_type k = 0; k < 3; ++k)
        {
          av += matrix[row, k] * vectors[k, col];
          norm += vectors[k, row] * vectors[k, col];
        }
        EXPECT_NEAR(av, (values[col] * vectors[row, col]), tolerance);
        EXPECT_NEAR(norm, row == col ? Scalar{1} : Scalar{0}, tolerance);
      }
    }
  }

  auto device_input = uni20::to_device(matrix, resources);
  auto selected = uni20::linalg::eigh(device_input, uni20::linalg::EigenvalueIndexRange{2, 3});
  static_assert(std::is_same_v<decltype(selected.eigenvectors), uni20::CudaMatrix<Scalar, uni20::ColumnMajor>>);
  auto selected_values = uni20::to_host(selected.eigenvalues);
  auto selected_vectors = uni20::to_host(selected.eigenvectors);
  ASSERT_EQ(selected_values.size(), 1);
  ASSERT_EQ(selected_vectors.extent(0), 3);
  ASSERT_EQ(selected_vectors.extent(1), 1);
  EXPECT_NEAR(selected_values[0], expected[2], tolerance);
  for (uni20::index_type row = 0; row < 3; ++row)
  {
    Scalar av = 0;
    for (uni20::index_type k = 0; k < 3; ++k)
      av += matrix[row, k] * selected_vectors[k, 0];
    EXPECT_NEAR(av, (selected_values[0] * selected_vectors[row, 0]), tolerance);
  }

  uni20::Tensor<Scalar, 2, uni20::HostStorage, uni20::ColumnMajor> work_host(3, 3);
  uni20::copy(work_host, matrix);
  auto work = uni20::to_device(work_host, resources);
  uni20::CudaTensor<Scalar, 1> device_values(uni20::uninitialized, resources, 3);
  uni20::linalg::self_adjoint_eigh(device_values, work, {.compute_vectors = false});
  auto values = uni20::to_host(device_values);
  for (uni20::index_type i = 0; i < 3; ++i)
    EXPECT_NEAR(values[i], expected[static_cast<std::size_t>(i)], tolerance);
}

} // namespace
