#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/linalg/backends/cusolver/svd.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

int cuda_device_count()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
  return count;
}

template <class Scalar> using real_t = uni20::make_real_t<Scalar>;

template <class Scalar> constexpr bool is_complex_v = !std::same_as<Scalar, real_t<Scalar>>;

struct Shape
{
    uni20::index_type rows = 0;
    uni20::index_type cols = 0;
};

// Square, tall and wide shapes, including vectors and sizes past small-matrix paths.
std::vector<Shape> const shapes = {{1, 1}, {4, 4}, {6, 3}, {3, 6}, {1, 5}, {5, 1}, {8, 5}, {5, 8}, {70, 33}, {33, 70}};

std::vector<double> known_singular_values(std::size_t rank)
{
  std::vector<double> values(rank);
  for (std::size_t k = 0; k < rank; ++k)
    values[k] = 0.5 + 1.25 * static_cast<double>(rank - 1 - k);
  return values;
}

// Dense Householder reflector I - 2 u u^H / (u^H u). Complex entries make the
// reflector genuinely complex, so the product below is not a real matrix in disguise.
std::vector<std::complex<double>> householder(std::size_t size, double seed, bool complex_entries)
{
  std::vector<std::complex<double>> u(size);
  double norm = 0.0;
  for (std::size_t k = 0; k < size; ++k)
  {
    double const imag = complex_entries ? 0.25 * static_cast<double>(static_cast<int>(k % 3) - 1) : 0.0;
    u[k] = {seed + static_cast<double>(k) * 0.5 - 0.125 * static_cast<double>(k % 2), imag};
    norm += std::norm(u[k]);
  }
  std::vector<std::complex<double>> reflector(size * size);
  for (std::size_t row = 0; row < size; ++row)
    for (std::size_t col = 0; col < size; ++col)
      reflector[row * size + col] = (row == col ? 1.0 : 0.0) - 2.0 * u[row] * std::conj(u[col]) / norm;
  return reflector;
}

// A = H_rows * diag(values) * H_cols, so the singular values of A are exactly `values`.
template <class Scalar, class Layout>
uni20::Tensor<Scalar, 2, uni20::HostStorage, Layout> make_known_matrix(Shape shape, std::vector<double> const& values)
{
  auto const rows = static_cast<std::size_t>(shape.rows);
  auto const cols = static_cast<std::size_t>(shape.cols);
  auto const left = householder(rows, 1.0, is_complex_v<Scalar>);
  auto const right = householder(cols, 2.0, is_complex_v<Scalar>);
  uni20::Tensor<Scalar, 2, uni20::HostStorage, Layout> matrix(shape.rows, shape.cols);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      std::complex<double> entry = 0.0;
      for (std::size_t k = 0; k < values.size(); ++k)
        entry += left[row * rows + k] * values[k] * right[k * cols + col];
      auto const r = static_cast<uni20::index_type>(row);
      auto const c = static_cast<uni20::index_type>(col);
      if constexpr (is_complex_v<Scalar>)
        matrix[r, c] = Scalar(static_cast<real_t<Scalar>>(entry.real()), static_cast<real_t<Scalar>>(entry.imag()));
      else
        matrix[r, c] = static_cast<Scalar>(entry.real());
    }
  }
  return matrix;
}

template <class Scalar> double tolerance(Shape shape, std::vector<double> const& values)
{
  double const largest = values.empty() ? 0.0 : values.front();
  double const extent = static_cast<double>(std::max(shape.rows, shape.cols));
  return 64.0 * static_cast<double>(std::numeric_limits<real_t<Scalar>>::epsilon()) * extent * largest;
}

template <class ValueTensor> int device_ordinal(ValueTensor const& values)
{
  return uni20::mdspec_of(values).data_descriptor().buffer().device().ordinal();
}

template <class Scalar, class ValueTensor>
void expect_known_values(ValueTensor const& device_values, Shape shape, std::vector<double> const& expected)
{
  static_assert(std::same_as<ValueTensor, uni20::CudaTensor<real_t<Scalar>, 1>>);
  auto const host_values = uni20::to_host(device_values);
  ASSERT_EQ(host_values.extent(0), static_cast<uni20::index_type>(expected.size()));
  double const tol = tolerance<Scalar>(shape, expected);
  for (std::size_t k = 0; k < expected.size(); ++k)
    EXPECT_NEAR(static_cast<double>(host_values[static_cast<uni20::index_type>(k)]), expected[k], tol) << "k=" << k;
}

template <class HostMatrix, class DeviceMatrix>
void expect_unchanged(HostMatrix const& expected, DeviceMatrix const& actual)
{
  auto const host = uni20::to_host(actual);
  for (uni20::index_type row = 0; row < expected.extent(0); ++row)
    for (uni20::index_type col = 0; col < expected.extent(1); ++col)
      EXPECT_EQ((host[row, col]), (expected[row, col]));
}

std::string describe(Shape shape, char const* layout)
{
  return std::to_string(shape.rows) + "x" + std::to_string(shape.cols) + " " + layout;
}

template <class Scalar> class CusolverSingularValuesTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      if (cuda_device_count() == 0) GTEST_SKIP() << "no CUDA devices are available";
    }
};

using SingularValueScalars = ::testing::Types<float, double, uni20::complex<float>, uni20::complex<double>>;
TYPED_TEST_SUITE(CusolverSingularValuesTest, SingularValueScalars);

TYPED_TEST(CusolverSingularValuesTest, PreservingCallStaysOnDeviceForAllShapesAndLayouts)
{
  using scalar_type = TypeParam;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  for (Shape const shape : shapes)
  {
    auto const expected = known_singular_values(static_cast<std::size_t>(std::min(shape.rows, shape.cols)));
    {
      SCOPED_TRACE(describe(shape, "column-major"));
      auto const host = make_known_matrix<scalar_type, uni20::ColumnMajor>(shape, expected);
      auto const device = uni20::to_device(host, resources);
      auto const values = uni20::linalg::singular_values(device);
      EXPECT_EQ(device_ordinal(values), 0);
      expect_known_values<scalar_type>(values, shape, expected);
      expect_unchanged(host, device);
    }
    {
      SCOPED_TRACE(describe(shape, "row-major"));
      auto const host = make_known_matrix<scalar_type, uni20::RowMajor>(shape, expected);
      auto const device = uni20::to_device(host, resources);
      auto const values = uni20::linalg::singular_values(device);
      expect_known_values<scalar_type>(values, shape, expected);
      expect_unchanged(host, device);
    }
  }
}

TYPED_TEST(CusolverSingularValuesTest, ConsumingCallUsesOrCopiesDeviceWorkspace)
{
  using scalar_type = TypeParam;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  for (Shape const shape : shapes)
  {
    auto const expected = known_singular_values(static_cast<std::size_t>(std::min(shape.rows, shape.cols)));
    {
      // Tall column-major input is destroyed in place; wide column-major input is copied to row-major work.
      SCOPED_TRACE(describe(shape, "column-major"));
      auto device = uni20::to_device(make_known_matrix<scalar_type, uni20::ColumnMajor>(shape, expected), resources);
      auto const values = uni20::linalg::singular_values(std::move(device));
      expect_known_values<scalar_type>(values, shape, expected);
    }
    {
      // Wide row-major input is destroyed in place as its tall transpose.
      SCOPED_TRACE(describe(shape, "row-major"));
      auto device = uni20::to_device(make_known_matrix<scalar_type, uni20::RowMajor>(shape, expected), resources);
      auto const values = uni20::linalg::singular_values(std::move(device));
      expect_known_values<scalar_type>(values, shape, expected);
    }
  }
}

TYPED_TEST(CusolverSingularValuesTest, AsyncFormsReturnDeviceValues)
{
  using scalar_type = TypeParam;
  using matrix_type = uni20::CudaMatrix<scalar_type>;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);

  for (Shape const shape : {Shape{6, 3}, Shape{3, 6}})
  {
    SCOPED_TRACE(describe(shape, "column-major"));
    auto const expected = known_singular_values(static_cast<std::size_t>(std::min(shape.rows, shape.cols)));
    auto const host = make_known_matrix<scalar_type, uni20::ColumnMajor>(shape, expected);

    uni20::async::Async<matrix_type> preserved = uni20::to_device(host, resources);
    auto preserved_values = uni20::linalg::singular_values(preserved);
    static_assert(
        std::same_as<decltype(preserved_values), uni20::async::Async<uni20::CudaTensor<real_t<scalar_type>, 1>>>);
    expect_known_values<scalar_type>(preserved_values.get_wait(scheduler), shape, expected);
    expect_unchanged(host, preserved.get_wait(scheduler));

    uni20::async::Async<matrix_type> consumed = uni20::to_device(host, resources);
    auto consumed_values = uni20::linalg::singular_values(std::move(consumed));
    static_assert(
        std::same_as<decltype(consumed_values), uni20::async::Async<uni20::CudaTensor<real_t<scalar_type>, 1>>>);
    expect_known_values<scalar_type>(consumed_values.get_wait(scheduler), shape, expected);
  }
}

class CusolverSingularValuesKernelTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      device_count_ = cuda_device_count();
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    int device_count_ = 0;
};

TEST_F(CusolverSingularValuesKernelTest, ExplicitSelectorComputesValuesOnDevice)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  Shape const shape{5, 8};
  auto const expected = known_singular_values(5);
  auto const device = uni20::to_device(make_known_matrix<double, uni20::ColumnMajor>(shape, expected), resources);
  auto const values = uni20::linalg::singular_values(uni20::linalg::CusolverBackend{}, device);
  expect_known_values<double>(values, shape, expected);
}

TEST_F(CusolverSingularValuesKernelTest, EmptyMatricesReturnEmptyDeviceValues)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  for (Shape const shape : {Shape{0, 3}, Shape{3, 0}, Shape{0, 0}})
  {
    SCOPED_TRACE(describe(shape, "column-major"));
    uni20::CudaMatrix<double> matrix(resources, shape.rows, shape.cols);
    auto const preserved = uni20::linalg::singular_values(matrix);
    EXPECT_EQ(preserved.extent(0), 0);
    auto const consumed = uni20::linalg::singular_values(std::move(matrix));
    EXPECT_EQ(consumed.extent(0), 0);
  }
}

TEST_F(CusolverSingularValuesKernelTest, KernelDeclinesLayoutsWhoseProviderShapeIsWide)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  uni20::CudaTensor<double, 1> values(resources, 3);

  uni20::CudaMatrix<double, uni20::ColumnMajor> wide_column_major(resources, 3, 6);
  auto values_spec = uni20::mdspec_of(values);
  auto wide_spec = uni20::mdspec_of(wide_column_major);
  EXPECT_EQ(uni20::linalg::try_kernel(uni20::linalg::CusolverBackend{}, uni20::linalg::singular_values_op{},
                                      values_spec, wide_spec),
            uni20::linalg::KernelAttempt::unsupported_layout);

  uni20::CudaMatrix<double, uni20::RowMajor> tall_row_major(resources, 6, 3);
  auto tall_spec = uni20::mdspec_of(tall_row_major);
  EXPECT_EQ(uni20::linalg::try_kernel(uni20::linalg::CusolverBackend{}, uni20::linalg::singular_values_op{},
                                      values_spec, tall_spec),
            uni20::linalg::KernelAttempt::unsupported_layout);
}

TEST_F(CusolverSingularValuesKernelTest, ValuesStayOnTheInputDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  auto& device1 = runtime.device_resources(1);
  Shape const shape{8, 5};
  auto const expected = known_singular_values(5);
  auto const matrix = uni20::to_device(make_known_matrix<double, uni20::ColumnMajor>(shape, expected), device1);

  auto const values = uni20::linalg::singular_values(matrix);
  EXPECT_EQ(device_ordinal(values), 1);
  expect_known_values<double>(values, shape, expected);
}

// The calling thread's current device is 0; provider calls must select the operand device.
TEST_F(CusolverSingularValuesKernelTest, FullSvdRunsOnTheOperandDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  auto& device1 = runtime.device_resources(1);
  Shape const shape{8, 5};
  auto const expected = known_singular_values(5);
  auto matrix = uni20::to_device(make_known_matrix<double, uni20::ColumnMajor>(shape, expected), device1);
  uni20::CudaTensor<double, 1> values(device1, 5);
  uni20::CudaMatrix<double> left(device1, 8, 5);
  uni20::CudaMatrix<double> right_adjoint(device1, 5, 5);

  uni20::linalg::singular_value_decomposition(uni20::linalg::CusolverBackend{}, values, left, right_adjoint, matrix);
  expect_known_values<double>(values, shape, expected);
}

TEST_F(CusolverSingularValuesKernelTest, KernelRejectsValuesOnAnotherDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  uni20::CudaTensor<double, 1> values(runtime.device_resources(1), 3);
  uni20::CudaMatrix<double> matrix(runtime.device_resources(0), 6, 3);
  auto values_spec = uni20::mdspec_of(values);
  auto matrix_spec = uni20::mdspec_of(matrix);
  EXPECT_EQ(uni20::linalg::try_kernel(uni20::linalg::CusolverBackend{}, uni20::linalg::singular_values_op{},
                                      values_spec, matrix_spec),
            uni20::linalg::KernelAttempt::incompatible_devices);
}

} // namespace
