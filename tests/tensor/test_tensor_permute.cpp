#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include "../linalg/deferred_host_tensor.hpp"

#include <gtest/gtest.h>

namespace
{
using matrix = uni20::DenseMatrix<double>;
using matrix_view = decltype(uni20::permute<1, 0>(std::declval<matrix&>()));

template <class T, std::size_t... Axis>
concept CanPermute = requires(T&& tensor) { uni20::permute<Axis...>(std::forward<T>(tensor)); };

static_assert(uni20::MutableImmediateTensorView<matrix_view>);
static_assert(!uni20::MutableTensorView<matrix_view const>);
static_assert(!uni20::OwningTensor<matrix_view>);
static_assert(CanPermute<matrix&, 1, 0>);
static_assert(CanPermute<matrix const&, 1, 0>);
static_assert(!CanPermute<matrix, 1, 0>);
static_assert(!CanPermute<matrix const, 1, 0>);
static_assert(CanPermute<matrix_view, 1, 0>);
static_assert(!CanPermute<matrix&, 0, 0>);
static_assert(!CanPermute<matrix&, 0, 2>);
static_assert(!CanPermute<matrix&, 0>);
static_assert(!CanPermute<matrix&, 0, 1, 2>);
using generator = decltype(uni20::ones<double>(2, 3));
static_assert(!CanPermute<generator&, 1, 0>);

template <class Layout> void check_rank_three()
{
  uni20::Tensor<double, 3, uni20::HostStorage, Layout> tensor(2, 3, 4);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        tensor[i, j, k] = 100 * i + 10 * j + k;
  auto view = uni20::permute<2, 0, 1>(tensor);
  EXPECT_EQ(view.mdspan().data_handle(), tensor.mdspan().data_handle());
  EXPECT_EQ(view.extent(0), 4);
  EXPECT_EQ(view.extent(1), 2);
  EXPECT_EQ(view.extent(2), 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        EXPECT_EQ((view[k, i, j]), (tensor[i, j, k]));
  view[3, 1, 2] = -17;
  EXPECT_EQ((tensor[1, 2, 3]), -17);

  auto packed = uni20::make_tensor(view);
  EXPECT_NE(packed.mdspan().data_handle(), tensor.mdspan().data_handle());
  EXPECT_TRUE(packed.mdspan().is_exhaustive());
  EXPECT_EQ((packed[3, 1, 2]), -17);
  view[3, 1, 2] = 5;
  EXPECT_EQ((packed[3, 1, 2]), -17);
}
} // namespace

TEST(TensorPermuteTest, ColumnAndRowMajorViewsAliasAndMaterialize)
{
  check_rank_three<uni20::ColumnMajor>();
  check_rank_three<uni20::RowMajor>();
}

TEST(TensorPermuteTest, PaddedStridesAndStaticExtentsArePreserved)
{
  using extents = stdex::extents<uni20::index_type, 2, stdex::dynamic_extent, 4>;
  using tensor_type = uni20::BasicTensor<double, extents, uni20::HostStorage, stdex::layout_stride>;
  tensor_type tensor(extents{3}, std::array<uni20::index_type, 3>{1, 5, 20});
  tensor[1, 2, 3] = 42;
  auto view = uni20::permute<2, 0, 1>(tensor);
  using permuted_extents = typename decltype(view)::extents_type;
  static_assert(permuted_extents::static_extent(0) == 4);
  static_assert(permuted_extents::static_extent(1) == 2);
  static_assert(permuted_extents::static_extent(2) == stdex::dynamic_extent);
  EXPECT_EQ(view.stride(0), 20);
  EXPECT_EQ(view.stride(1), 1);
  EXPECT_EQ(view.stride(2), 5);
  EXPECT_EQ((view[3, 1, 2]), 42);
  EXPECT_FALSE(view.mdspan().is_exhaustive());
}

TEST(TensorPermuteTest, RankSixRegroupingNeedsNoSymmetryWrapper)
{
  uni20::Tensor<double, 6> tensor(2, 3, 2, 3, 2, 3);
  tensor[1, 2, 1, 2, 1, 2] = 123;
  tensor[0, 1, 0, 1, 0, 1] = 456;
  auto packed = uni20::make_tensor(uni20::permute<0, 3, 1, 2, 4, 5>(tensor));
  EXPECT_EQ((packed[1, 2, 2, 1, 1, 2]), 123);
  EXPECT_EQ((packed[0, 1, 1, 0, 0, 1]), 456);
  EXPECT_EQ(packed.extent(2), 3);
  EXPECT_EQ(packed.extent(3), 2);
}

TEST(TensorPermuteTest, TemporaryPermutationDescriptorsComposeWithoutRetainingIntermediates)
{
  uni20::Tensor<double, 3> tensor(2, 3, 4);
  tensor[1, 2, 3] = 42;
  auto roundtrip = uni20::permute<1, 2, 0>(uni20::permute<2, 0, 1>(tensor));
  EXPECT_EQ(roundtrip.mdspan().data_handle(), tensor.mdspan().data_handle());
  EXPECT_EQ((roundtrip[1, 2, 3]), 42);
  roundtrip[1, 2, 3] = 12;
  EXPECT_EQ((tensor[1, 2, 3]), 12);
}

TEST(TensorPermuteTest, ConstnessAndConjugatingAccessorArePreserved)
{
  uni20::DenseMatrix<uni20::complex<double>> tensor(2, 3);
  tensor[1, 2] = {4, 7};
  auto const_view = uni20::permute<1, 0>(std::as_const(tensor));
  static_assert(!uni20::MutableTensorView<decltype(const_view)>);
  auto conjugated = uni20::conj(tensor);
  auto view = uni20::permute<1, 0>(conjugated);
  static_assert(!uni20::MutableTensorView<decltype(view)>);
  static_assert(uni20::mdspan_needs_conjugation_v<decltype(view.mdspan())>);
  EXPECT_EQ((view[2, 1]), (uni20::complex<double>{4, -7}));
  auto packed = uni20::make_tensor(view);
  EXPECT_EQ((packed[2, 1]), (uni20::complex<double>{4, -7}));
}

TEST(TensorPermuteTest, IdentityScalarEmptyAndSingletonAxes)
{
  uni20::ScalarTensor<double> scalar;
  scalar[] = 9;
  auto scalar_view = uni20::permute<>(scalar);
  EXPECT_EQ(scalar_view[], 9);
  auto identity = uni20::permute<>(scalar.mdspan());
  static_assert(std::same_as<decltype(identity), decltype(scalar.mdspan())>);
  matrix empty(0, 3);
  auto empty_view = uni20::permute<1, 0>(empty);
  EXPECT_EQ(empty_view.extent(0), 3);
  EXPECT_EQ(empty_view.extent(1), 0);
  auto empty_copy = uni20::make_tensor(empty_view);
  EXPECT_EQ(empty_copy.extent(0), 3);
  EXPECT_EQ(empty_copy.extent(1), 0);
  uni20::Tensor<double, 3> empty_cube(0, 3, 4);
  auto empty_cube_view = uni20::permute<2, 0, 1>(empty_cube);
  EXPECT_EQ(empty_cube_view.extent(0), 4);
  EXPECT_EQ(empty_cube_view.extent(1), 0);
  EXPECT_EQ(empty_cube_view.extent(2), 3);
  // Positive strides alone are insufficient: layout_stride also requires
  // a dimension ordering whose neighboring storage intervals do not overlap.
  EXPECT_GE(empty_cube_view.stride(1), empty_cube_view.stride(0) * empty_cube_view.extent(0));
  EXPECT_GE(empty_cube_view.stride(2), empty_cube_view.stride(1) * empty_cube_view.extent(1));
  uni20::Tensor<double, 3> tensor(2, 1, 3);
  tensor[1, 0, 2] = 8;
  auto singleton_view = uni20::permute<1, 2, 0>(tensor);
  EXPECT_EQ((singleton_view[0, 2, 1]), 8);
}

TEST(TensorPermuteTest, DeferredViewsRetainDescriptorsAndSelector)
{
  uni20::test::DeferredHostTensor<double, 2> tensor(2, 3);
  auto source = tensor.mdspec();
  auto view = uni20::permute<1, 0>(tensor);
  static_assert(!uni20::ImmediateTensorView<decltype(view)>);
  static_assert(uni20::MutableTensorView<decltype(view)>);
  static_assert(
      std::same_as<std::remove_cvref_t<decltype(view.backend_selector())>, decltype(tensor.backend_selector())>);
  EXPECT_EQ(view.mdspec().data_descriptor().storage, source.data_descriptor().storage);
  EXPECT_EQ(view.mdspec().data_descriptor().element_offset, source.data_descriptor().element_offset);
  auto permuted_metadata = view.mdspec();
  auto lease = uni20::test::acquire_host_write_access_sync(permuted_metadata);
  lease.mdspan()[2, 1] = 13;
  auto source_lease = uni20::test::acquire_host_read_access_sync(std::as_const(tensor).mdspec());
  EXPECT_EQ((source_lease.mdspan()[1, 2]), 13);
}

TEST(TensorPermuteTest, StatefulBackendSelectorSurvivesNestedViews)
{
  struct Selector
  {
      int value = 0;
  };
  struct SelectedMatrix : matrix
  {
      using matrix::matrix;
      Selector selected{17};
      auto backend_selector() const { return selected; }
  };
  SelectedMatrix tensor(2, 3);
  auto view = uni20::permute<1, 0>(uni20::permute<1, 0>(tensor));
  EXPECT_EQ(view.backend_selector().value, 17);
}

#if UNI20_HAS_MPREAL
TEST(TensorPermuteTest, PrecisionBindingSurvivesIntermediateViewsAndTracksOwner)
{
  auto check = []<class S>() {
    auto p = uni20::Precision::bits(192), q = uni20::Precision::bits(80);
    uni20::DenseMatrix<S> tensor(2, 3, p);
    tensor[1, 2] = S(7, p);
    auto view = uni20::permute<1, 0>(tensor);
    auto nested = uni20::permute<1, 0>(uni20::permute<1, 0>(tensor));
    tensor.default_precision(q);
    EXPECT_EQ(view.default_precision(), q);
    EXPECT_EQ(nested.default_precision(), q);
    EXPECT_EQ((view[2, 1].precision()), p);
    auto copy = uni20::make_tensor(view);
    EXPECT_EQ(copy.default_precision(), q);
    EXPECT_EQ((copy[2, 1]), S(7, p));
    tensor.default_precision(p);
    EXPECT_EQ(view.default_precision(), p);
    EXPECT_EQ(copy.default_precision(), q);
    uni20::DenseMatrix<S> empty(0, 3, p);
    auto empty_view = uni20::permute<1, 0>(empty);
    empty.default_precision(q);
    EXPECT_EQ(empty_view.default_precision(), q);
    uni20::DenseMatrix<S> unset;
    auto unset_view = uni20::permute<1, 0>(unset);
    EXPECT_FALSE(unset_view.default_precision_if_set());
    EXPECT_THROW(static_cast<void>(unset_view.default_precision()), std::logic_error);
    unset.default_precision(uni20::Precision::exact());
    EXPECT_EQ(unset_view.default_precision(), uni20::Precision::exact());
  };
  check.template operator()<uni20::mpreal>();
  check.template operator()<uni20::complex<uni20::mpreal>>();
}
#endif
