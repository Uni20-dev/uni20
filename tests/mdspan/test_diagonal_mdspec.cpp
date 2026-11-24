#include <uni20/mdspan/diagonal_accessor.hpp>
#include <uni20/mdspan/diagonal_view.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{

using namespace uni20;

TEST(DiagonalMdspecTest, PresentsStridedComponentsAsAFullLogicalTensor)
{
  using component_extents_type = stdex::dextents<index_type, 1>;
  using component_mapping_type = uni20::layout_signed_stride::mapping<component_extents_type>;
  using component_mdspan_type = uni20::mdspan<double, component_extents_type, uni20::layout_signed_stride>;
  using logical_extents_type = stdex::dextents<index_type, 2>;

  std::array<double, 6> storage{10.0, 0.0, 20.0, 0.0, 30.0, 0.0};
  auto const component_mapping = component_mapping_type{component_extents_type{3}, std::array<index_type, 1>{-2}};
  component_mdspan_type components{storage.data() + 4, component_mapping};
  auto diagonal = make_diagonal_mdspan(logical_extents_type{3, 4}, components);

  static_assert(DiagonalMdspecLike<decltype(diagonal)>);
  static_assert(MutableDiagonalMdspecLike<decltype(diagonal)>);
  static_assert(!StridedMdspecLike<decltype(diagonal)>);
  EXPECT_EQ(diagonal.extent(0), 3);
  EXPECT_EQ(diagonal.extent(1), 4);
  EXPECT_DOUBLE_EQ((diagonal[0, 0]), 30.0);
  EXPECT_DOUBLE_EQ((diagonal[1, 1]), 20.0);
  EXPECT_DOUBLE_EQ((diagonal[2, 2]), 10.0);
  EXPECT_DOUBLE_EQ((diagonal[0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((diagonal[2, 3]), 0.0);

  diagonal[1, 1] = 25.0;
  EXPECT_DOUBLE_EQ(storage[2], 25.0);
  diagonal[0, 3] = 0.0;

  auto observed_components = diagonal_components(diagonal);
  static_assert(MutableRankedMdspanLike<decltype(observed_components), 1>);
  static_assert(AffineMdspanLike<decltype(observed_components)>);
  static_assert(!StridedMdspanLike<decltype(observed_components)>);
  EXPECT_EQ(observed_components.extent(0), 3);
  EXPECT_EQ(observed_components.stride(0), -2);
  EXPECT_EQ(observed_components.data_handle(), storage.data() + 4);
  EXPECT_DOUBLE_EQ(observed_components[0], 30.0);
  EXPECT_DOUBLE_EQ(observed_components[1], 25.0);
  EXPECT_DOUBLE_EQ(observed_components[2], 10.0);

  auto const_diagonal = make_const_mdspan(diagonal);
  static_assert(DiagonalMdspecLike<decltype(const_diagonal)>);
  static_assert(!MutableDiagonalMdspecLike<decltype(const_diagonal)>);
  auto const_components = diagonal_components(const_diagonal);
  static_assert(std::same_as<decltype(const_components)::layout_type, layout_signed_stride>);
  static_assert(!MutableMdspecLike<decltype(const_components)>);
  EXPECT_DOUBLE_EQ(const_components[1], 25.0);
}

TEST(DiagonalMdspecTest, RejectsTheWrongNumberOfComponents)
{
  using component_extents_type = stdex::dextents<index_type, 1>;
  using logical_extents_type = stdex::dextents<index_type, 3>;
  std::array<double, 1> storage{};
  uni20::mdspan<double, component_extents_type> components{storage.data(), component_extents_type{1}};

  EXPECT_THROW(static_cast<void>(make_diagonal_mdspan(logical_extents_type{2, 3, 4}, components)),
               std::invalid_argument);
}

TEST(DiagonalMdspecTest, PreservesPhysicalStridesWithNarrowLogicalIndices)
{
  using logical_shape = stdex::extents<std::uint8_t, 2, 2>;
  using component_shape = stdex::extents<int, 2>;
  std::array<int, 257> storage{};
  storage[0] = 11;
  storage[256] = 23;
  uni20::layout_stride::mapping<component_shape> mapping{component_shape{}, std::array{256}};
  uni20::mdspan<int, component_shape, uni20::layout_stride> components{storage.data(), mapping};

  auto diagonal = make_diagonal_mdspan(logical_shape{}, components);
  EXPECT_EQ((diagonal[1, 1]), 23);
  auto recovered = diagonal_components(diagonal);
  static_assert(std::same_as<decltype(recovered)::index_type, int>);
  static_assert(std::same_as<decltype(recovered)::layout_type, layout_stride>);
  auto const_components = diagonal_components(make_const_mdspan(diagonal));
  static_assert(std::same_as<decltype(const_components)::index_type, int>);
  static_assert(std::same_as<decltype(const_components)::layout_type, layout_stride>);
  EXPECT_EQ(const_components[1], 23);
  EXPECT_EQ(recovered.stride(0), 256);
  recovered[1] = 37;
  EXPECT_EQ(storage[256], 37);
  EXPECT_EQ(storage[0], 11);
}

TEST(DiagonalMdspecTest, RejectsComponentSpanThatCannotBeRecoveredInItsIndexType)
{
  using shape = uni20::extents<std::int8_t, 2, 2>;
  using mapping = uni20::detail::DiagonalLayout::mapping<shape>;
  int storage[128]{};
  auto square = uni20::mdspan{storage, mapping{shape{}, {64, 63}}};
  auto source = uni20::detail::diagonal_mdspec<0, 1>(square);
  ASSERT_EQ(source.mapping().required_span_size(), 128U);
  EXPECT_THROW(static_cast<void>(make_diagonal_mdspan(shape{}, source)), std::overflow_error);
}

struct ShiftedSignedComponentLayout
{
    template <class Extents> struct mapping : layout_signed_stride::mapping<Extents>
    {
        using base_type = layout_signed_stride::mapping<Extents>;
        using layout_type = ShiftedSignedComponentLayout;

        constexpr mapping() noexcept : base_type(Extents{}, {-2}) {}
        constexpr int operator()(int i) const noexcept { return 1 + base_type::operator()(i); }
        constexpr std::size_t base_offset() const noexcept { return 1; }
    };
};

TEST(DiagonalMdspecTest, BorrowsComponentsFromTheirMappedOrigin)
{
  using shape = uni20::extents<int, 2>;
  using mapping = ShiftedSignedComponentLayout::mapping<shape>;
  std::array<int, 4> storage{5, 10, 20, 30};
  auto components = uni20::mdspan{storage.data() + 2, mapping{}};
  auto diagonal = make_diagonal_mdspan(uni20::extents<int, 2, 2>{}, components);
  auto recovered = diagonal_components(diagonal);
  EXPECT_EQ(recovered.data_handle(), storage.data() + 3);
  EXPECT_EQ((diagonal[0, 0]), 30);
  EXPECT_EQ((diagonal[1, 1]), 10);
}

TEST(DiagonalMdspecTest, PreservesLinearLogicalOffsetsWithNarrowExtents)
{
  std::array<int, 17> storage{};
  storage.back() = 53;
  uni20::mdspan<int, stdex::extents<int, 17>> components{storage.data()};
  auto diagonal = make_diagonal_mdspan(stdex::extents<std::uint8_t, 17, 17>{}, components);
  EXPECT_EQ((diagonal[16, 16]), 53);
  EXPECT_EQ((diagonal[16, 15]), 0);
}

TEST(DiagonalMdspecTest, ComparesComponentCountsBeforeNarrowing)
{
  std::array<int, 258> storage{};
  uni20::mdspan<int, stdex::extents<int, 258>> components{storage.data()};
  using logical_shape = stdex::extents<std::uint8_t, 2, 2>;
  EXPECT_THROW((void)make_diagonal_mdspan(logical_shape{}, components), std::invalid_argument);
}

TEST(DiagonalMdspecTest, PreservesFullUnsignedSingletonComponentStride)
{
  using shape = stdex::extents<std::size_t, 1>;
  int storage = 42;
  uni20::layout_stride::mapping<shape> mapping{shape{}, std::array{std::numeric_limits<std::size_t>::max()}};
  uni20::mdspan<int, shape, uni20::layout_stride> components{&storage, mapping};
  auto diagonal = make_diagonal_mdspan(shape{}, components);
  auto recovered = diagonal_components(diagonal);
  EXPECT_EQ(recovered.stride(0), std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(recovered[0], 42);
}

TEST(DiagonalMdspecTest, LargeLogicalOffsetWith32BitComponents)
{
  constexpr std::int32_t count = 50000;
  using logical_shape = stdex::extents<std::int32_t, count, count>;
  std::vector<int> values(count);
  values.back() = 31;
  uni20::mdspan<int, stdex::extents<std::int32_t, count>> components{values.data()};
  auto diagonal = make_diagonal_mdspan(logical_shape{}, components);
  EXPECT_EQ(diagonal.mapping()(count - 1, count - 1), 2499999999LL);
  EXPECT_EQ((diagonal[count - 1, count - 1]), 31);
  static_assert(std::same_as<decltype(diagonal_components(diagonal))::index_type, std::int32_t>);
}

TEST(DiagonalMdspecTest, ChecksFlattenedLogicalSizeAtConstruction)
{
  using shape = stdex::dextents<std::ptrdiff_t, 3>;
  auto const large = std::numeric_limits<std::ptrdiff_t>::max();
  int value = 7;
  uni20::mdspan<int, stdex::extents<int, 1>> one{&value};
  EXPECT_THROW((void)make_diagonal_mdspan(shape{large, 2, 1}, one), std::overflow_error);

  uni20::mdspan<int, stdex::extents<int, 0>> empty{&value};
  auto diagonal = make_diagonal_mdspan(shape{large, 2, 0}, empty);
  EXPECT_EQ(diagonal.mapping().required_span_size(), 0);
}

TEST(DiagonalMdspecTest, ExposesAlignedOffsetAndSteppedComponents)
{
  using original_extents_type = stdex::dextents<index_type, 2>;
  using view_extents_type = stdex::dextents<index_type, 2>;
  using mapping_type = uni20::layout_stride::mapping<view_extents_type>;
  using accessor_type = diagonal_accessor<double, original_extents_type>;
  using view_type = uni20::mdspan<double, view_extents_type, uni20::layout_stride, accessor_type>;

  std::array<double, 5> storage{2.0, 3.0, 5.0, 7.0, 11.0};
  auto const mapping = mapping_type{view_extents_type{2, 2}, std::array<index_type, 2>{10, 2}};
  view_type aligned{diagonal_data_handle<double>{storage.data(), 6}, mapping,
                    accessor_type{original_extents_type{5, 5}}};

  EXPECT_DOUBLE_EQ((aligned[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((aligned[1, 1]), 7.0);
  auto components = diagonal_components(aligned);
  EXPECT_EQ(components.extent(0), 2);
  EXPECT_EQ(components.stride(0), 2);
  EXPECT_EQ(components.data_handle(), storage.data() + 1);
  EXPECT_DOUBLE_EQ(components[0], 3.0);
  EXPECT_DOUBLE_EQ(components[1], 7.0);

  view_type misaligned{diagonal_data_handle<double>{storage.data(), 1}, mapping,
                       accessor_type{original_extents_type{5, 5}}};
  EXPECT_THROW(static_cast<void>(diagonal_components(misaligned)), std::invalid_argument);
}

} // namespace
