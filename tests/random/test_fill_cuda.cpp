#include <gtest/gtest.h>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/mdspan/layout_signed_stride.hpp>
#include <uni20/random/fill.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

namespace uni20::random::test
{
namespace
{
class DenseRandomCudaTest : public ::testing::Test {
    void SetUp() override
    {
      int n = 0;
      auto status = cudaGetDeviceCount(&n);
      if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && n == 0))
        GTEST_SKIP() << "No CUDA device available";
      ASSERT_EQ(status, cudaSuccess);
    }
};

template <typename Base, typename Mapping> auto subview(Base base, std::size_t handle, Mapping mapping)
{
  using View = mdspec<typename Base::element_type, typename Mapping::extents_type, typename Mapping::layout_type,
                      typename Base::accessor_type, typename Base::data_descriptor_type>;
  return View{base.data_descriptor().offset_by(handle), mapping, base.accessor()};
}

TEST_F(DenseRandomCudaTest, PaddedReverseHonoursExplicitOrStandaloneOrigin)
{
  auto runtime = cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  Tensor<float, 1> zeros(18);
  auto allocation = to_device(zeros, 0);
  auto leaf = child(root(1), 2);
  using E = dextents<index_type, 2>;
  layout_signed_stride::mapping<E> map{E{2, 3}, std::array<index_type, 2>{-8, 2}};
  auto out = subview(allocation.mdspec(), 2, map);
  for (unsigned origin : {0u, 2u})
  {
    if (origin == 0)
      random::fill(out, leaf, Uniform<float>{});
    else
      random::fill(out, leaf, Uniform<float>{}, origin);
    auto result = to_host(allocation);
    for (unsigned i = 0; i < 18; ++i)
    {
      bool touched = i == 2 || i == 4 || i == 6 || i == 10 || i == 12 || i == 14;
      EXPECT_EQ(result[i], touched ? value_at(leaf, origin + i - 2, Uniform<float>{}) : 0.f);
    }
  }
}

TEST_F(DenseRandomCudaTest, CpuAndGpuCanSplitAnAllocationAtAnyLane)
{
  auto runtime = cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  auto leaf = child(root(9), 4);
  constexpr int n = 19;
  Tensor<double, 1> whole(n);
  random::fill(whole, leaf, Uniform<double>{});
  for (int split : {0, 1, 4, 13, n})
  {
    Tensor<double, 1> partial(n);
    auto head = mdspan{partial.mdspan().data_handle(), dextents<index_type, 1>{split}};
    random::fill(head, leaf, Uniform<double>{}, 0);
    auto allocation = to_device(partial, 0);
    using E = dextents<index_type, 1>;
    auto tail = subview(allocation.mdspec(), split, layout_right::mapping<E>{E{n - split}});
    random::fill(tail, leaf, Uniform<double>{}, split);
    auto result = to_host(allocation);
    for (int i = 0; i < n; ++i)
      EXPECT_EQ(result[i], whole[i]);
  }
}

TEST_F(DenseRandomCudaTest, RealAndComplexNormalTensorFillsUseEveryLane)
{
  auto runtime = cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  auto leaf = root(5);
  CudaTensor<double, 1> real(runtime.device_resources(0), 9);
  CudaTensor<complex<double>, 1> z(runtime.device_resources(0), 9);
  random::fill(real, leaf, Normal{0, 2});
  random::fill(z, leaf, ComplexNormal{{1, -1}, 2});
  auto r = to_host(real);
  auto c = to_host(z);
  for (int i = 0; i < 9; ++i)
  {
    double x = value_at(leaf, i, Normal{0, 2});
    auto y = value_at(leaf, i, ComplexNormal{{1, -1}, 2});
    double eps = std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(r[i], x, 8 * eps * std::abs(x));
    EXPECT_NEAR(c[i].real(), y.real(), 8 * eps * (1 + std::abs(y.real() - 1)));
    EXPECT_NEAR(c[i].imag(), y.imag(), 8 * eps * (1 + std::abs(y.imag() + 1)));
  }
}
} // namespace
} // namespace uni20::random::test
