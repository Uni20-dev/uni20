#include <gtest/gtest.h>
#include <uni20/linalg/ops/scale_columns.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/diagonal.hpp>
#include <uni20/tensor/permute.hpp>

namespace
{
using namespace uni20;
using namespace uni20::linalg;

auto fixture_matrix()
{
  DenseMatrix<double> a(3, 2);
  for (index_type j = 0; j < 2; ++j)
    for (index_type i = 0; i < 3; ++i)
      a[i, j] = 1 + 3 * j + i;
  return a;
}
auto fixture_diagonal()
{
  DenseMatrix<double> d(3, 3);
  d[0, 0] = -2;
  d[1, 1] = 0;
  d[2, 2] = 3;
  return d;
}
void check_result(DenseMatrix<double> const& a)
{
  // The permuted matrix has two rows and three columns. Its factors were the strided diagonal.
  for (index_type j = 0; j < 2; ++j)
  {
    EXPECT_EQ((a[0, j]), -2 * (1 + 3 * j));
    EXPECT_EQ((a[1, j]), 0);
    EXPECT_EQ((a[2, j]), 3 * (3 + 3 * j));
  }
}
TEST(ScaleColumns, RectangularStridedFactors)
{
  auto a = fixture_matrix();
  auto d = fixture_diagonal();
  scale_columns(backend_list{CpuReferenceBackend{}}, permute<1, 0>(a), diagonal_view<0, 1>(d));
  check_result(a);
  DenseMatrix<double> empty(0, 3);
  scale_columns(empty, diagonal_view<0, 1>(d));
  EXPECT_THROW(scale_columns(a, (diagonal_view<0, 1>(d))), std::invalid_argument);
}
#if UNI20_BACKEND_CUDA
TEST(ScaleColumns, CudaRectangularStridedFactors)
{
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !devices))
    GTEST_SKIP();
  ASSERT_EQ(status, cudaSuccess);
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  auto a = to_device(fixture_matrix(), 0);
  auto d = to_device(fixture_diagonal(), 0);
  scale_columns(backend_list{CudaReferenceBackend{}}, permute<1, 0>(a), diagonal_view<0, 1>(d));
  check_result(to_host(a));
}
#endif
} // namespace
