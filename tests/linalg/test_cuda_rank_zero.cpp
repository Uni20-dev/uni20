// Rank-zero tensors in the canonical (column-major) layout, which standard
// mdspan does not give a callable stride(), on CUDA storage.
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/reductions.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>

namespace
{

using complex_type = uni20::complex<double>;
using host_scalar = uni20::ScalarTensor<double>;
using cuda_scalar = uni20::CudaTensor<double, 0>;
using complex_host_scalar = uni20::ScalarTensor<complex_type>;
using complex_cuda_scalar = uni20::CudaTensor<complex_type, 0>;

static_assert(uni20::StridedMdspecLike<uni20::tensor_mdspec_t<cuda_scalar>>);
static_assert(uni20::StridedMdspecLike<uni20::tensor_mdspec_t<host_scalar>>);

class CudaRankZeroTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess) GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    int device_count_ = 0;
};

TEST_F(CudaRankZeroTest, ScalarTensorsTransferBetweenHostAndDevice)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  host_scalar host;
  host[] = 2.5;
  auto device = uni20::to_device(host, runtime.device_resources(0));
  EXPECT_EQ(uni20::to_host(device)[], 2.5);

  complex_host_scalar complex_host;
  complex_host[] = complex_type{1.0, -2.0};
  auto complex_device = uni20::to_device(complex_host, runtime.device_resources(0));
  EXPECT_EQ(uni20::to_host(complex_device)[], (complex_type{1.0, -2.0}));

  host_scalar written;
  uni20::copy(written, device);
  EXPECT_EQ(written[], 2.5);
}

TEST_F(CudaRankZeroTest, ScalarTensorsCopyBetweenDeviceTensors)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  host_scalar host;
  host[] = -7.0;
  auto source = uni20::to_device(host, runtime.device_resources(0));
  cuda_scalar target(uni20::uninitialized, runtime.device_resources(0), cuda_scalar::extents_type{});
  uni20::copy(target, source);
  EXPECT_EQ(uni20::to_host(target)[], -7.0);
}

TEST_F(CudaRankZeroTest, ScalarTensorsCopyBetweenDevices)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  host_scalar host;
  host[] = 4.0;
  auto source = uni20::to_device(host, runtime.device_resources(0));
  cuda_scalar target(uni20::uninitialized, runtime.device_resources(1), cuda_scalar::extents_type{});
  uni20::copy(target, source);
  EXPECT_EQ(target.storage().device().ordinal(), 1);
  EXPECT_EQ(uni20::to_host(target)[], 4.0);
}

#if UNI20_BACKEND_CUBLAS
TEST_F(CudaRankZeroTest, ScalarTensorsReduceToHostScalars)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  host_scalar host;
  host[] = -3.0;
  auto device = uni20::to_device(host, runtime.device_resources(0));
  EXPECT_DOUBLE_EQ(uni20::norm_host(device), 3.0);
  EXPECT_DOUBLE_EQ(uni20::inner_product_host(device, device), 9.0);

  complex_host_scalar lhs;
  lhs[] = complex_type{1.0, 2.0};
  complex_host_scalar rhs;
  rhs[] = complex_type{3.0, -1.0};
  auto device_lhs = uni20::to_device(lhs, runtime.device_resources(0));
  auto device_rhs = uni20::to_device(rhs, runtime.device_resources(0));
  EXPECT_DOUBLE_EQ(uni20::norm_host(device_lhs), std::sqrt(5.0));
  EXPECT_EQ(uni20::inner_product_host(device_lhs, device_rhs), std::conj(lhs[]) * rhs[]);
}
#endif

} // namespace
