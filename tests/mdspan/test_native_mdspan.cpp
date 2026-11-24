#include <uni20/mdspan/mdspec.hpp>
#include <uni20/mdspan/reverse.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace
{

using MatrixShape = uni20::dextents<int, 2>;
using ScalarShape = uni20::extents<int>;

template <class Shape, class Accessor>
concept SupportsSignedLayout = requires {
  typename uni20::mdspan<int, Shape, uni20::layout_signed_stride, Accessor>;
  typename uni20::mdspec<int, Shape, uni20::layout_signed_stride, Accessor, int>;
};

struct UnsignedOffsetAccessor : uni20::default_accessor<int>
{
    using offset_type = std::size_t;
};

static_assert(SupportsSignedLayout<MatrixShape, uni20::default_accessor<int>>);
static_assert(!SupportsSignedLayout<uni20::dextents<unsigned, 2>, uni20::default_accessor<int>>);
static_assert(!SupportsSignedLayout<MatrixShape, UnsignedOffsetAccessor>);

TEST(NativeMdspanTest, ConfiguredExtentsPreserveMixedShape)
{
  using Shape = uni20::extents<int, 2, uni20::dynamic_extent, 4, uni20::dynamic_extent>;
  using WideShape = uni20::extents<std::ptrdiff_t, 2, uni20::dynamic_extent, 4, uni20::dynamic_extent>;
  static_assert(Shape::rank() == 4 && Shape::rank_dynamic() == 2);
  static_assert(ScalarShape::rank() == 0 && ScalarShape::rank_dynamic() == 0);
  static_assert(std::same_as<Shape::index_type, int> && std::same_as<Shape::size_type, unsigned>);
#if UNI20_USE_STD_EXTENTS
  static_assert(std::same_as<Shape, std::extents<int, 2, std::dynamic_extent, 4, std::dynamic_extent>>);
#else
  static_assert(std::is_empty_v<ScalarShape> && std::is_empty_v<uni20::extents<int, 2, 3>>);
#endif
  constexpr std::array dynamic{3, 5};
  constexpr std::array all{2, 3, 4, 5};
  constexpr Shape mixed{dynamic};
  static_assert(mixed == Shape{all} && mixed == Shape{std::span{dynamic}});
  static_assert(WideShape{mixed} == mixed && Shape{WideShape{mixed}} == mixed);
  static_assert(Shape{}.extent(1) == 0 && Shape{}.extent(2) == 4);
  static_assert(mixed.extent(0) == 2 && mixed.extent(1) == 3 && mixed.extent(2) == 4 && mixed.extent(3) == 5);
}

template <class Span> constexpr auto axis_steps(Span const& span)
{
  typename Span::index_type steps = 0;
  for (std::size_t axis = 0; axis < Span::rank(); ++axis)
    steps += span.stride(axis);
  return steps;
}

TEST(NativeMdspanTest, DeducesArrayExtentWithoutChangingPointerDeduction)
{
  int const data[]{1, 2, 3, 4, 5, 6};
  auto array = uni20::mdspan{data};
  static_assert(std::same_as<typename decltype(array)::element_type, int const>);
  static_assert(decltype(array)::rank() == 1 && decltype(array)::static_extent(0) == 6);
  EXPECT_EQ(array[5], 6);

  auto const* pointer = data;
  auto scalar = uni20::mdspan{pointer};
  static_assert(decltype(scalar)::rank() == 0);
  EXPECT_EQ(scalar[], 1);
  EXPECT_EQ((uni20::mdspan{data, 2, 3}[1, 2]), 6);
}

TEST(NativeMdspanTest, ConvertsStridedMappingsWithCanonicalStrides)
{
  using Left = uni20::layout_left::mapping<MatrixShape>;
  using Right = uni20::layout_right::mapping<MatrixShape>;
  using Strided = uni20::layout_stride::mapping<MatrixShape>;
  using ScalarStrided = uni20::layout_stride::mapping<ScalarShape>;
  using OtherScalarStrided = uni20::layout_stride::mapping<uni20::extents<unsigned int>>;
  using ScalarLeft = uni20::layout_left::mapping<ScalarShape>;
  using ScalarRight = uni20::layout_right::mapping<ScalarShape>;
  static_assert(std::is_convertible_v<ScalarStrided, ScalarLeft>);
  static_assert(std::is_convertible_v<ScalarStrided, ScalarRight>);
  static_assert(std::is_constructible_v<ScalarRight, OtherScalarStrided>);
  static_assert(!std::is_convertible_v<OtherScalarStrided, ScalarRight>);
  static_assert(!std::is_convertible_v<Strided, Left>);
  static_assert(!std::is_convertible_v<Strided, Right>);

  constexpr Left left = MatrixShape{2, 3};
  constexpr Right right = MatrixShape{2, 3};
  static_assert(Left{Strided{left}} == left);
  static_assert(Right{Strided{right}} == right);

  int values[6]{1, 2, 3, 4, 5, 6};
  auto scalar_strided = uni20::mdspan{values, ScalarStrided{}};
  uni20::mdspan<int, ScalarShape, uni20::layout_right> const scalar = scalar_strided;
  EXPECT_EQ(scalar[], 1);

  auto strided = uni20::mdspan{values, Strided{left}};
  uni20::mdspan<int, MatrixShape, uni20::layout_left> const contiguous{strided};
  EXPECT_EQ((contiguous[0, 1]), 3);
}

TEST(NativeMdspanTest, ContiguousIndexingUsesTheLayout)
{
  int data[]{0, 1, 2, 3, 4, 5};
  auto right = uni20::mdspan{data, MatrixShape{2, 3}};
  auto left = uni20::mdspan{data, uni20::layout_left::mapping<MatrixShape>{MatrixShape{2, 3}}};

  static_assert(uni20::MutableStridedMdspanLike<decltype(right)>);
  static_assert(std::same_as<typename decltype(right)::offset_type, std::ptrdiff_t>);
  EXPECT_EQ((right[0, 1]), 1);
  EXPECT_EQ((left[0, 1]), 2);
  EXPECT_EQ(right.stride(0), 3);
  EXPECT_EQ(left.stride(0), 1);
  EXPECT_EQ(right.size(), 6U);
  static_assert(decltype(right)::mapping_type::is_base_offset_always_zero());
  EXPECT_EQ(right.mapping().base_offset(), 0U);
  EXPECT_EQ(right.mapping().required_span_size(), 6);
  EXPECT_EQ((uni20::mdspan{data, 2, 3}[1, 2]), 5);
  EXPECT_EQ((uni20::mdspan{data, std::array{2, 3}}[1, 2]), 5);

  auto copy = uni20::mdspan{right};
  uni20::mdspan<int const, MatrixShape> const_view = copy;
  static_assert(!uni20::MutableMdspanLike<decltype(const_view)>);
  EXPECT_EQ(const_view.data_handle(), data);

  auto read_only = uni20::make_const_mdspan(right);
  static_assert(std::same_as<decltype(read_only), decltype(const_view)>);
  EXPECT_EQ((read_only[0, 1]), 1);

  using WideShape = uni20::dextents<std::ptrdiff_t, 2>;
  uni20::mdspan<int const, WideShape, uni20::layout_stride> strided = left;
  EXPECT_EQ(strided.mapping().strides(), (std::array<std::ptrdiff_t, 2>{1, 2}));
  EXPECT_EQ((strided[0, 1]), 2);
}

TEST(NativeMdspanTest, ComparesStridedAndContiguousMappings)
{
  using ScalarStrided = uni20::layout_stride::mapping<ScalarShape>;
  using ScalarLeft = uni20::layout_left::mapping<ScalarShape>;
  using ScalarRight = uni20::layout_right::mapping<ScalarShape>;
  static_assert(ScalarStrided{} == ScalarLeft{} && ScalarLeft{} == ScalarStrided{});
  static_assert(ScalarStrided{} == ScalarRight{} && ScalarRight{} == ScalarStrided{});

  using Left = uni20::layout_left::mapping<MatrixShape>;
  using Right = uni20::layout_right::mapping<MatrixShape>;
  using Strided = uni20::layout_stride::mapping<MatrixShape>;
  constexpr Left left{MatrixShape{2, 3}};
  constexpr Strided strided{left};
  static_assert(strided == left && left == strided);
  using UnsignedStrided = uni20::layout_stride::mapping<uni20::dextents<unsigned int, 2>>;
  constexpr UnsignedStrided mixed_width{uni20::dextents<unsigned int, 2>{2, 3}, {1, 2}};
  static_assert(strided == mixed_width && mixed_width == strided);
  static_assert(strided != Right{left.extents()});
  static_assert(strided != Left{MatrixShape{3, 2}});

  struct ShiftedMapping : Left
  {
      using Left::Left;

      constexpr int operator()(int i, int j) const noexcept { return 1 + Left::operator()(i, j); }
      constexpr int required_span_size() const noexcept
      {
        return Left::required_span_size() == 0 ? 0 : 1 + Left::required_span_size();
      }
  };
  static_assert(strided != ShiftedMapping{left.extents()});
  constexpr Left empty{MatrixShape{0, 3}};
  static_assert(Strided{empty} == ShiftedMapping{empty.extents()});
}

TEST(NativeMdspanTest, RankZeroHasAnObserverAndOneElement)
{
  int data = 42;
  uni20::layout_left::mapping<ScalarShape> mapping{};
  auto view = uni20::mdspan{&data, mapping};
  uni20::mdspec<int, ScalarShape, uni20::layout_left, uni20::default_accessor<int>, int> descriptor{0, mapping, {}};

  static_assert(uni20::StridedMdspanLike<decltype(view)>);
  static_assert(uni20::StridedMdspecLike<decltype(descriptor)>);
  static_assert(decltype(view)::is_base_offset_always_zero());
  static_assert(decltype(descriptor)::is_base_offset_always_zero());
  static_assert(uni20::layout_stride::mapping<ScalarShape>::is_always_exhaustive());
  EXPECT_EQ(axis_steps(view), 0);
  EXPECT_EQ(axis_steps(descriptor), 0);
  EXPECT_EQ(view[], 42);
  EXPECT_EQ(view.size(), 1U);
  EXPECT_FALSE(view.empty());
  EXPECT_EQ(view.mapping().base_offset(), 0U);
  EXPECT_EQ(view.mapping().required_span_size(), 1);

#ifdef __cpp_lib_mdspan
  std::mdspan<int, std::extents<int>, std::layout_stride> foreign{&data, {}};
  auto borrowed = uni20::mdspan{foreign};
  EXPECT_EQ(borrowed[], 42);
  EXPECT_EQ(axis_steps(borrowed), 0);
#endif
}

TEST(NativeMdspanTest, InterleavedDiagonalStridesAndZeroSingletonStride)
{
  using layout = uni20::layout_stride;
  int data[12]{};
  auto view = uni20::mdspan{data, layout::mapping<MatrixShape>{MatrixShape{3, 2}, {4, 3}}};
  view[2, 1] = 42;
  EXPECT_EQ(data[11], 42);
  EXPECT_EQ(view.size(), 6U);
  EXPECT_EQ(view.mapping().required_span_size(), 12);
  EXPECT_EQ(view.mapping().base_offset(), 0U);
  EXPECT_FALSE(view.is_exhaustive());

  layout::mapping<MatrixShape> const singleton{MatrixShape{1, 3}, {0, 1}};
  EXPECT_TRUE(singleton.is_exhaustive());
  EXPECT_EQ(singleton(0, 2), 2);
}

TEST(NativeMdspanTest, EmptyMappingKeepsItsStrides)
{
  using Shape = uni20::extents<int, 0, 3, 4>;
  using Strided = uni20::layout_stride::mapping<Shape>;
  constexpr uni20::layout_left::mapping<Shape> left{};
  constexpr Strided strided = left;
  constexpr uni20::mdspan<int, Shape, uni20::layout_stride> view{nullptr, strided};
  static_assert(Strided::is_always_exhaustive());
  static_assert(view.empty() && view.size() == 0);
  static_assert(strided.is_exhaustive() && strided.required_span_size() == 0);
  static_assert(strided.base_offset() == 0);
  static_assert(strided.strides() == std::array{1, 0, 0});
}

TEST(NativeMdspanTest, SignedMappingsDescribeStorageOnBothSidesOfTheHandle)
{
  using layout = uni20::layout_signed_stride;
  using Span = uni20::mdspan<int, MatrixShape, layout>;
  using Spec = uni20::mdspec<int, MatrixShape, layout, uni20::default_accessor<int>, int>;
  static_assert(!Span::is_base_offset_always_zero());
  static_assert(!Spec::is_base_offset_always_zero());
  constexpr layout::mapping<MatrixShape> compact{MatrixShape{2, 3}, {-1, 2}};
  static_assert(!decltype(compact)::is_base_offset_always_zero());
  static_assert(std::same_as<decltype(compact.base_offset()), std::size_t>);
  static_assert(std::same_as<decltype(compact.required_span_size()), std::size_t>);
  static_assert(compact.is_unique() && compact.is_exhaustive());
  static_assert(compact.base_offset() == 1 && compact.required_span_size() == 6);
  static_assert(compact(1, 2) == 3);

  constexpr layout::mapping<MatrixShape> padded{MatrixShape{3, 2}, {-3, 2}};
  static_assert(padded.base_offset() == 6 && padded.required_span_size() == 9);
  static_assert(!padded.is_exhaustive());
  constexpr layout::mapping<MatrixShape> empty{MatrixShape{3, 0}, {-1, 4}};
  static_assert(empty.base_offset() == 0 && empty.required_span_size() == 0 && empty.is_exhaustive());
  constexpr layout::mapping<ScalarShape> scalar{};
  static_assert(scalar.base_offset() == 0 && scalar.required_span_size() == 1 && scalar.is_exhaustive());

  using narrow_shape = uni20::extents<std::int8_t, 2, 2>;
  constexpr layout::mapping<narrow_shape> wide_span{narrow_shape{}, {-128, 127}};
  static_assert(wide_span.base_offset() == 128 && wide_span.required_span_size() == 256);
}

TEST(NativeMdspanTest, ReverseRetainsAnEmptyHandleAndItsStrides)
{
  using shape = uni20::extents<int, 0, 3>;
  constexpr uni20::mdspan<int, shape, uni20::layout_left> source{nullptr, shape{}};
  constexpr auto reversed = uni20::reverse<1>(source);
  static_assert(reversed.empty());
  static_assert(reversed.mapping().base_offset() == 0 && reversed.mapping().required_span_size() == 0);
  static_assert(reversed.data_handle() == nullptr && reversed.mapping().strides() == std::array{1, 0});
}

#ifdef __cpp_lib_mdspan
TEST(NativeMdspanTest, ImportBorrowsStorageAndPreservesStaticExtents)
{
  using Shape = std::extents<std::int16_t, 2, std::dynamic_extent>;
  using Source = std::mdspan<int const, Shape, std::layout_left>;
  int data[]{0, 1, 2, 3, 4, 5};
  Source const source{data, 3};
  auto imported = uni20::mdspan{source};

  static_assert(
      std::same_as<
          decltype(imported),
          uni20::mdspan<int const, uni20::extents<std::ptrdiff_t, 2, uni20::dynamic_extent>, uni20::layout_left>>);
  static_assert(!std::is_convertible_v<Source, decltype(imported)>);
  static_assert(!std::is_constructible_v<uni20::mdspan<int, uni20::extents<std::int16_t, 2, uni20::dynamic_extent>,
                                                      uni20::layout_left>, Source>);
  EXPECT_EQ(imported.data_handle(), data);
  EXPECT_EQ(imported.stride(1), 2);
  EXPECT_EQ((imported[1, 2]), 5);

  struct CustomAccessor : std::default_accessor<int const>
  {};
  using CustomView = std::mdspan<int const, Shape, std::layout_left, CustomAccessor>;
  static_assert(!std::is_constructible_v<decltype(imported), CustomView>);
}

TEST(NativeMdspanTest, ImportChecksRequiredSpanSizeBeforeNarrowing)
{
  using Shape = std::dextents<std::int16_t, 1>;
  using Source = std::mdspan<int, Shape, std::layout_stride>;
  using Narrow = uni20::mdspan<int, uni20::dextents<std::int8_t, 1>, uni20::layout_stride>;
  int data[128]{};
  auto source = [&](int extent, int stride) {
    return Source{data, Source::mapping_type{Shape{extent}, std::array{stride}}};
  };

  auto fits = Narrow{source(2, 126)};
  EXPECT_EQ(fits.mapping().required_span_size(), 127);
  EXPECT_THROW(static_cast<void>(Narrow{source(2, 127)}), std::overflow_error);
  EXPECT_THROW(static_cast<void>(Narrow{source(1, 128)}), std::overflow_error);
  EXPECT_EQ(uni20::mdspan{source(2, 127)}.mapping().required_span_size(), 128);
}

TEST(NativeMdspanTest, ImportChecksShapeEvenWhenEmpty)
{
  using Shape = std::dextents<std::int16_t, 3>;
  using Narrow = uni20::mdspan<int, uni20::dextents<std::int8_t, 3>>;
  using Fixed = uni20::mdspan<int, uni20::extents<int, 1, uni20::dynamic_extent, uni20::dynamic_extent>>;
  std::mdspan<int, Shape> empty{nullptr, 0, 16, 16};
  EXPECT_THROW(static_cast<void>(Narrow{empty}), std::overflow_error);
  EXPECT_THROW(static_cast<void>(Fixed{empty}), std::invalid_argument);

  using Source = std::mdspan<int, std::dextents<std::size_t, 1>, std::layout_stride>;
  int data = 0;
  Source singleton{&data,
                   Source::mapping_type{Source::extents_type{1}, std::array{std::numeric_limits<std::size_t>::max()}}};
  EXPECT_THROW(static_cast<void>(uni20::mdspan{singleton}), std::overflow_error);
}

TEST(NativeMdspanTest, ImportStandardMdspan)
{
  using Shape = std::extents<int, 3, std::dynamic_extent>;
  int data[12]{};
  std::mdspan<int, Shape, std::layout_stride> source{data,
                                                     std::layout_stride::mapping<Shape>{Shape{2}, std::array{4, 1}}};
  auto view = uni20::mdspan{source};
  static_assert(decltype(view)::static_extent(0) == 3);
  view[2, 1] = 7;
  EXPECT_EQ(data[9], 7);
  EXPECT_EQ(view.stride(0), 4);
}
#endif

} // namespace
