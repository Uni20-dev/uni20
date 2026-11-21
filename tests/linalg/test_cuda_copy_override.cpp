// User `backend_selector_override` specializations for `copy_op`, declared after
// every public header, as docs/architecture/kernel_dispatch.md describes. The
// headers must not instantiate these overrides first. This file is its own test
// executable so that no other translation unit sees different selectors for the
// same operation and storage policy.
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <concepts>

template <> struct uni20::linalg::backend_selector_override<uni20::linalg::copy_op, uni20::HostStorage>
{
    static auto select(uni20::linalg::copy_op const&) { return uni20::linalg::CpuReferenceBackend{}; }
};

template <> struct uni20::linalg::backend_selector_override<uni20::linalg::copy_op, uni20::CudaStorage>
{
    static auto select(uni20::linalg::copy_op const&) { return uni20::CudaStorage::backend_selector(); }
};

namespace
{

static_assert(
    std::same_as<decltype(uni20::linalg::select_backend_for_storage<uni20::HostStorage>(uni20::linalg::copy_op{})),
                 uni20::linalg::CpuReferenceBackend>);
static_assert(
    std::same_as<decltype(uni20::linalg::select_backend_for_storage<uni20::CudaStorage>(uni20::linalg::copy_op{})),
                 uni20::CudaStorage::backend_selector_type>);

TEST(CudaCopyOverrideTest, StagedTransfersUseOverriddenCopySelectors)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    GTEST_SKIP() << "no CUDA devices are available";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});

  uni20::RowMajorTensor<double, 2> host(2, 3);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 3; ++column)
      host[row, column] = 10.0 * row + column;
  }
  uni20::CudaTensor<double, 2> device(runtime.device_resources(0), 2, 3);
  uni20::copy(device, host);
  uni20::RowMajorTensor<double, 2> back(2, 3);
  uni20::copy(back, device);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_EQ((back[row, column]), (host[row, column]));
  }
}

} // namespace
