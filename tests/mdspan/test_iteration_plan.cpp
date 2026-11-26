#include "../helpers.hpp"
#include "gtest/gtest.h"
#include <uni20/mdspan/iteration_plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>

using namespace uni20;

TEST(MakeIterationPlanTest, SimpleContiguousPlan)
{
  auto mapping = make_mapping(std::array<std::size_t, 1>{10}, std::array<index_t, 1>{1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(plan[0].extent, 10);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, MergedContiguousPlan)
{
  auto mapping = make_mapping(std::array<std::size_t, 3>{10, 20, 30}, std::array<index_t, 3>{1, 10, 200});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 10 * 20 * 30);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, ZeroExtentProducesRetainedZeroDim)
{
  // A zero-extent dim means 0 elements. It is carried as a single retained
  // extent-0 dim -- NOT an empty plan, which denotes a rank-0 scalar (1
  // element). The driver then loops that dim zero times.
  auto mapping = make_mapping(std::array<std::size_t, 1>{0}, std::array<index_t, 1>{1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 0);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, OutOfOrderStrides)
{
  auto mapping = make_mapping(std::array<std::size_t, 3>{30, 20, 10}, std::array<index_t, 3>{200, 10, 1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 10 * 20 * 30);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, InnerNegativeStride)
{
  auto mapping = make_mapping(std::array<std::size_t, 1>{10}, std::array<index_t, 1>{-1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(plan[0].extent, 10);
  EXPECT_EQ(offset, -9);
}

TEST(MakeIterationPlanTest, OuterNegativeStride)
{
  auto mapping = make_mapping(std::array<std::size_t, 2>{4, 2}, std::array<index_t, 2>{-8, 1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 2);
  EXPECT_EQ(plan[0].extent, 4);
  EXPECT_EQ(plan[0].stride, 8);
  EXPECT_EQ(offset, -24);
}

TEST(MakeIterationPlanTest, NegativeStrideMiddleDimension)
{
  auto mapping = make_mapping(std::array<std::size_t, 3>{4, 3, 2}, std::array<index_t, 3>{1, -4, 20});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_EQ(plan.size(), 2);
  EXPECT_EQ(offset, -8);
}

TEST(MakeIterationPlanTest, MixedSignsNoMerge)
{
  auto mapping = make_mapping(std::array<std::size_t, 2>{4, 3}, std::array<index_t, 2>{-7, 1});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_EQ(plan.size(), 2);    // Can't merge
  EXPECT_EQ(plan[0].extent, 4); // Outer
  EXPECT_EQ(plan[0].stride, 7);
  EXPECT_EQ(plan[1].extent, 3); // Inner
  EXPECT_EQ(plan[1].stride, 1);
  EXPECT_EQ(offset, -21); // offset = (extent-1)*stride = (4-1)*-7 = -21
}

TEST(MakeIterationPlanTest, MergeableNegativeStrides)
{
  auto mapping = make_mapping(std::array<std::size_t, 2>{4, 5}, std::array<index_t, 2>{-1, -4});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 20);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(offset, -19);
}

TEST(MakeIterationPlanTest, SizeOneDimsAreDropped)
{
  // Size-1 surviving dims contribute index 0 only, so they are dropped from the
  // plan; only the extent-10 dim survives.
  auto mapping = make_mapping(std::array<std::size_t, 3>{1, 10, 1}, std::array<index_t, 3>{999, 1, 7});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(plan.size(), 1);
  EXPECT_EQ(plan[0].extent, 10);
  EXPECT_EQ(plan[0].stride, 1);
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, AllSizeOneIsScalarEmptyPlan)
{
  // Every dim size-1 => rank-0 scalar => empty plan (which the driver reads as
  // exactly one element, never zero).
  auto mapping = make_mapping(std::array<std::size_t, 3>{1, 1, 1}, std::array<index_t, 3>{5, 9, 2});
  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_TRUE(plan.empty());
  EXPECT_EQ(offset, 0);
}

TEST(MakeIterationPlanTest, RankZeroMdspanIsScalar)
{
  using extents_t = uni20::dextents<index_t, 0>;
  using mapping_t = uni20::layout_stride::mapping<extents_t>;

  std::array<std::ptrdiff_t, 0> strides{};
  mapping_t mapping{extents_t{}, strides};

  auto [plan, offset] = make_iteration_plan_with_offset(mapping);
  EXPECT_TRUE(plan.empty());
  EXPECT_EQ(offset, 0);
}

TEST(IterationPlanTest, NormalizesMinimumNarrowStrideWithoutNarrowing)
{
  using shape = uni20::extents<std::int8_t, 2>;
  uni20::layout_signed_stride::mapping<shape> const mapping{shape{}, {-128}};
  auto [dimensions, offset] = make_iteration_plan_with_offset(mapping);
  ASSERT_EQ(dimensions.size(), 1U);
  EXPECT_EQ(dimensions[0].extent, 2);
  EXPECT_EQ(dimensions[0].stride, 128);
  EXPECT_EQ(offset, -128);
}

TEST(IterationPlanTest, PreservesEachMappingOrigin)
{
  using shape = uni20::extents<std::uint8_t, 2>;
  using base_mapping = uni20::layout_right::mapping<shape>;
  struct Mapping : base_mapping
  {
      constexpr int operator()(std::uint8_t i) const { return 300 + i; }
      constexpr int required_span_size() const { return 302; }
      static constexpr bool is_always_exhaustive() { return false; }
      static constexpr bool is_exhaustive() { return false; }
  };
  Mapping shifted;
  auto [dimensions, offset] = make_iteration_plan_with_offset(shifted);
  ASSERT_EQ(dimensions.size(), 1U);
  EXPECT_EQ(offset, shifted(0));
  auto plan = make_multi_iteration_plan(std::tuple{base_mapping{}, shifted});
  EXPECT_TRUE(plan.representable);
  EXPECT_EQ(plan.base_offsets[0], 0);
  EXPECT_EQ(plan.base_offsets[1], shifted(0));
  EXPECT_EQ(plan.reachable_offsets[1].maximum, shifted(1));
}

TEST(MakeMultiIterationPlanTest, SimpleMatchingLayouts)
{
  auto output = make_mapping(std::array<std::size_t, 2>{10, 2}, std::array<index_t, 2>{2, 1});
  auto input = make_mapping(std::array<std::size_t, 2>{10, 2}, std::array<index_t, 2>{20, 10});
  auto plan = make_multi_iteration_plan(std::tuple{output, input});

  ASSERT_EQ(plan.rank(), 1);
  EXPECT_EQ(plan.dimensions[0].extent, 20);
  EXPECT_EQ(plan.dimensions[0].strides[0], 1);
  EXPECT_EQ(plan.dimensions[0].strides[1], 10);
  EXPECT_EQ(plan.base_offsets[0], 0);
  EXPECT_EQ(plan.base_offsets[1], 0);
  EXPECT_EQ(plan.element_count, 20);
  EXPECT_EQ(plan.reachable_offsets[0].minimum, 0);
  EXPECT_EQ(plan.reachable_offsets[0].maximum, 19);
  EXPECT_EQ(plan.reachable_offsets[1].minimum, 0);
  EXPECT_EQ(plan.reachable_offsets[1].maximum, 190);
}

TEST(MakeMultiIterationPlanTest, RankZeroMappingProducesOneScalarElement)
{
  using extents_type = uni20::dextents<index_t, 0>;
  using mapping_type = uni20::layout_left::mapping<extents_type>;
  auto plan = make_multi_iteration_plan(std::tuple{mapping_type{extents_type{}}});

  EXPECT_EQ(plan.rank(), 0);
  EXPECT_FALSE(plan.empty());
  EXPECT_EQ(plan.element_count, 1);
  EXPECT_EQ(plan.base_offsets[0], 0);
  EXPECT_EQ(plan.reachable_offsets[0].minimum, 0);
  EXPECT_EQ(plan.reachable_offsets[0].maximum, 0);
}

TEST(MakeMultiIterationPlanTest, MismatchedButMergeable)
{
  auto output = make_mapping(std::array<std::size_t, 2>{3, 4}, std::array<index_t, 2>{4, 1});
  auto input = make_mapping(std::array<std::size_t, 2>{3, 4}, std::array<index_t, 2>{40, 10});
  auto plan = make_multi_iteration_plan(std::tuple{output, input});

  ASSERT_EQ(plan.rank(), 1);
  EXPECT_EQ(plan.dimensions[0].extent, 12);
  EXPECT_EQ(plan.dimensions[0].strides[0], 1);
  EXPECT_EQ(plan.dimensions[0].strides[1], 10);
  EXPECT_EQ(plan.base_offsets[0], 0);
  EXPECT_EQ(plan.base_offsets[1], 0);
  EXPECT_EQ(plan.element_count, 12);
}

TEST(MakeMultiIterationPlanTest, DimensionsMergeOnlyWhenEveryOperandIsAdjacent)
{
  auto output = make_mapping(std::array<std::size_t, 2>{2, 3}, std::array<index_t, 2>{3, 1});
  auto lhs = make_mapping(std::array<std::size_t, 2>{2, 3}, std::array<index_t, 2>{1, 2});
  auto rhs = make_mapping(std::array<std::size_t, 2>{2, 3}, std::array<index_t, 2>{10, 3});
  auto plan = make_multi_iteration_plan(std::tuple{output, lhs, rhs});

  ASSERT_EQ(plan.rank(), 2);
  EXPECT_EQ(plan.element_count, 6);
  EXPECT_EQ(plan.reachable_offsets[0].maximum, 5);
  EXPECT_EQ(plan.reachable_offsets[1].maximum, 5);
  EXPECT_EQ(plan.reachable_offsets[2].maximum, 16);
}

TEST(MakeMultiIterationPlanTest, MergeProbeDoesNotOverflowForExtremeInnerStride)
{
  auto const large_stride = std::numeric_limits<std::ptrdiff_t>::max() / 2 + 1;
  extent_strides<1> outer{2, std::array<std::ptrdiff_t, 1>{1}};
  extent_strides<1> inner{2, std::array<std::ptrdiff_t, 1>{large_stride}};

  EXPECT_FALSE(outer.can_merge_with_inner(inner));
}

TEST(ExtentStrideTest, SingletonDimensionMergesWithoutContributingItsStride)
{
  extent_stride<std::size_t, std::ptrdiff_t> outer{4, 7};
  extent_stride<std::size_t, std::ptrdiff_t> inner_singleton{1, 999};

  ASSERT_TRUE(outer.can_merge_with_inner(inner_singleton));
  outer.merge_with_inner(inner_singleton);
  EXPECT_EQ(outer.extent, 4);
  EXPECT_EQ(outer.stride, 7);

  extent_stride<std::size_t, std::ptrdiff_t> outer_singleton{1, -23};
  extent_stride<std::size_t, std::ptrdiff_t> inner{5, 3};

  ASSERT_TRUE(outer_singleton.can_merge_with_inner(inner));
  outer_singleton.merge_with_inner(inner);
  EXPECT_EQ(outer_singleton.extent, 5);
  EXPECT_EQ(outer_singleton.stride, 3);
}

TEST(MakeMultiIterationPlanTest, OutputNegativeStrideFlipsEveryOperand)
{
  auto output = make_mapping(std::array<std::size_t, 1>{5}, std::array<index_t, 1>{-2});
  auto input = make_mapping(std::array<std::size_t, 1>{5}, std::array<index_t, 1>{3});
  auto plan = make_multi_iteration_plan(std::tuple{output, input});

  ASSERT_EQ(plan.rank(), 1);
  EXPECT_EQ(plan.dimensions[0].extent, 5);
  EXPECT_EQ(plan.dimensions[0].strides[0], 2);
  EXPECT_EQ(plan.dimensions[0].strides[1], -3);
  EXPECT_EQ(plan.base_offsets[0], -8);
  EXPECT_EQ(plan.base_offsets[1], 12);
  EXPECT_EQ(plan.reachable_offsets[0].minimum, -8);
  EXPECT_EQ(plan.reachable_offsets[0].maximum, 0);
  EXPECT_EQ(plan.reachable_offsets[1].minimum, 0);
  EXPECT_EQ(plan.reachable_offsets[1].maximum, 12);
}

TEST(MakeMultiIterationPlanTest, MixedSignsPreventMerge)
{
  auto output = make_mapping(std::array<std::size_t, 2>{4, 2}, std::array<index_t, 2>{1, -4});
  auto input = make_mapping(std::array<std::size_t, 2>{4, 2}, std::array<index_t, 2>{10, 40});
  auto plan = make_multi_iteration_plan(std::tuple{output, input});

  ASSERT_EQ(plan.rank(), 2);
  EXPECT_EQ(plan.dimensions[0].extent, 2);
  EXPECT_EQ(plan.dimensions[0].strides[0], 4);
  EXPECT_EQ(plan.dimensions[0].strides[1], -40);
  EXPECT_EQ(plan.dimensions[1].extent, 4);
  EXPECT_EQ(plan.dimensions[1].strides[0], 1);
  EXPECT_EQ(plan.dimensions[1].strides[1], 10);
  EXPECT_EQ(plan.base_offsets[0], -4);
  EXPECT_EQ(plan.base_offsets[1], 40);
  EXPECT_EQ(plan.reachable_offsets[0].minimum, -4);
  EXPECT_EQ(plan.reachable_offsets[0].maximum, 3);
  EXPECT_EQ(plan.reachable_offsets[1].minimum, 0);
  EXPECT_EQ(plan.reachable_offsets[1].maximum, 70);
}

TEST(MakeMultiIterationPlanTest, ZeroExtentProducesRetainedZeroDim)
{
  auto output = make_mapping(std::array<std::size_t, 1>{0}, std::array<index_t, 1>{1});
  auto input = make_mapping(std::array<std::size_t, 1>{0}, std::array<index_t, 1>{3});
  auto plan = make_multi_iteration_plan(std::tuple{output, input});

  ASSERT_EQ(plan.rank(), 1);
  EXPECT_EQ(plan.dimensions[0].extent, 0);
  EXPECT_EQ(plan.base_offsets[0], 0);
  EXPECT_EQ(plan.base_offsets[1], 0);
  EXPECT_EQ(plan.element_count, 0);
  EXPECT_TRUE(plan.reachable_offsets[0].empty());
  EXPECT_TRUE(plan.reachable_offsets[1].empty());
}

TEST(IterationPlanTest, EmptyShapesDoNotQueryStrides)
{
  using extents_type = uni20::dextents<index_t, 2>;
  using base_mapping = uni20::layout_left::mapping<extents_type>;
  struct Mapping : base_mapping
  {
      using base_mapping::base_mapping;
      index_t stride(std::size_t axis) const
      {
        ADD_FAILURE() << "Empty iteration queried stride on axis " << axis;
        return base_mapping::stride(axis);
      }
  };

  for (auto shape : {extents_type{3, 0}, extents_type{0, 3}})
  {
    Mapping mapping{shape};
    auto [dimensions, offset] = make_iteration_plan_with_offset(mapping);
    ASSERT_EQ(dimensions.size(), 1);
    EXPECT_EQ(dimensions[0].extent, 0);
    EXPECT_EQ(offset, 0);

    auto plan = make_multi_iteration_plan(std::tuple{mapping, mapping});
    EXPECT_TRUE(plan.representable);
    EXPECT_TRUE(plan.empty());
  }
}

TEST(MakeMultiIterationPlanTest, TrailingZeroMakesIterationEmptyAfterALargePrefix)
{
  constexpr auto maximum = std::numeric_limits<index_t>::max();
  auto const extents = std::array<std::size_t, 3>{static_cast<std::size_t>(maximum), 2, 0};
  auto const strides = std::array<index_t, 3>{1, maximum, 1};
  auto output = make_mapping(extents, strides);
  auto input = make_mapping(extents, strides);
  auto plan = make_multi_iteration_plan(std::tuple{output, input});

  EXPECT_TRUE(plan.representable);
  ASSERT_EQ(plan.rank(), 1);
  EXPECT_EQ(plan.dimensions[0].extent, 0);
  EXPECT_EQ(plan.element_count, 0);
  EXPECT_TRUE(plan.reachable_offsets[0].empty());
  EXPECT_TRUE(plan.reachable_offsets[1].empty());
}

TEST(MakeMultiIterationPlanTest, AllSizeOneIsScalarEmptyPlan)
{
  auto output = make_mapping(std::array<std::size_t, 2>{1, 1}, std::array<index_t, 2>{7, 3});
  auto input = make_mapping(std::array<std::size_t, 2>{1, 1}, std::array<index_t, 2>{11, 5});
  auto plan = make_multi_iteration_plan(std::tuple{output, input});

  EXPECT_EQ(plan.rank(), 0);
  EXPECT_FALSE(plan.empty());
  EXPECT_EQ(plan.element_count, 1);
  EXPECT_EQ(plan.base_offsets[0], 0);
  EXPECT_EQ(plan.base_offsets[1], 0);
  EXPECT_EQ(plan.reachable_offsets[0].minimum, 0);
  EXPECT_EQ(plan.reachable_offsets[0].maximum, 0);
}
