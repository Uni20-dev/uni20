#include <gtest/gtest.h>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/mdspan/mdspec.hpp>
#include <uni20/mdspan/stride.hpp>

#include <array>
#include <cstddef>

namespace
{

using index_type = std::ptrdiff_t;
using scalar_extents = stdex::extents<index_type>;
using matrix_extents = stdex::dextents<index_type, 2>;

template <class Layout> using scalar_span = stdex::mdspan<double, scalar_extents, Layout>;
template <class Layout> using matrix_span = stdex::mdspan<double, matrix_extents, Layout>;

struct Region
{
    double* data = nullptr;
};

template <class Extents, class Layout>
using test_mdspec = uni20::mdspec<double, Extents, Layout, stdex::default_accessor<double>, Region>;

// Standard layout_left and layout_right constrain stride() away at rank zero;
// layout_stride keeps it.
template <class Mapping>
concept HasMemberStride = requires(Mapping const& mapping, std::size_t axis) { mapping.stride(axis); };
static_assert(!HasMemberStride<stdex::layout_left::mapping<scalar_extents>>);
static_assert(!HasMemberStride<stdex::layout_right::mapping<scalar_extents>>);
static_assert(HasMemberStride<stdex::layout_stride::mapping<scalar_extents>>);

// A rank-zero mapping is strided in every layout, and uni20::stride exists for it.
static_assert(uni20::StridedMdspanLike<scalar_span<stdex::layout_left>>);
static_assert(uni20::StridedMdspanLike<scalar_span<stdex::layout_right>>);
static_assert(uni20::StridedMdspanLike<scalar_span<stdex::layout_stride>>);
static_assert(uni20::StridedMdspecLike<test_mdspec<scalar_extents, stdex::layout_left>>);
static_assert(uni20::StridedMdspecLike<test_mdspec<scalar_extents, stdex::layout_right>>);
static_assert(uni20::StrideObservable<stdex::layout_left::mapping<scalar_extents>>);
static_assert(uni20::StrideObservable<scalar_span<stdex::layout_right>>);
static_assert(uni20::StrideObservable<test_mdspec<scalar_extents, stdex::layout_left>>);
static_assert(HasMemberStride<test_mdspec<scalar_extents, stdex::layout_left>>);

// At positive rank a mapping without stride() is neither strided nor stride-observable.
struct UnstridedMapping
{
    using index_type = std::ptrdiff_t;
    using extents_type = stdex::dextents<index_type, 1>;
};
struct UnstridedScalarMapping
{
    using index_type = std::ptrdiff_t;
    using extents_type = scalar_extents;
};
static_assert(!uni20::StrideObservable<UnstridedMapping>);
static_assert(uni20::StrideObservable<UnstridedScalarMapping>);
static_assert(uni20::StridedMdspanLike<matrix_span<stdex::layout_left>>);
static_assert(uni20::StridedMdspecLike<test_mdspec<matrix_extents, stdex::layout_right>>);

template <class Layout> void expect_matrix_strides_match_members()
{
  std::array<double, 12> storage{};
  matrix_span<Layout> span(storage.data(), 3, 4);
  test_mdspec<matrix_extents, Layout> spec{Region{storage.data()}, span.mapping(), {}};
  for (std::size_t axis = 0; axis < 2; ++axis)
  {
    EXPECT_EQ(uni20::stride(span, axis), span.stride(axis));
    EXPECT_EQ(uni20::stride(span.mapping(), axis), span.mapping().stride(axis));
    EXPECT_EQ(uni20::stride(spec, axis), span.stride(axis));
    EXPECT_EQ(spec.stride(axis), span.stride(axis));
  }
}

TEST(MdspanStride, MatchesTheMappingAtPositiveRank)
{
  expect_matrix_strides_match_members<stdex::layout_left>();
  expect_matrix_strides_match_members<stdex::layout_right>();

  std::array<double, 20> storage{};
  stdex::layout_stride::mapping<matrix_extents> const padded(matrix_extents{3, 4}, std::array<index_type, 2>{1, 5});
  matrix_span<stdex::layout_stride> span(storage.data(), padded);
  EXPECT_EQ(uni20::stride(span, 0), 1);
  EXPECT_EQ(uni20::stride(span, 1), 5);
}

// A loop over the axes of a rank-zero operand compiles and does nothing.
template <class Span> index_type sum_of_strides(Span const& span)
{
  index_type total = 0;
  for (std::size_t axis = 0; axis < Span::rank(); ++axis)
    total += uni20::stride(span, axis);
  return total;
}

TEST(MdspanStride, AxisLoopsCompileAtRankZero)
{
  double value = 1.0;
  EXPECT_EQ(sum_of_strides(scalar_span<stdex::layout_left>(&value)), 0);
  EXPECT_EQ(sum_of_strides(scalar_span<stdex::layout_right>(&value)), 0);
  EXPECT_EQ(sum_of_strides(test_mdspec<scalar_extents, stdex::layout_left>{Region{&value}, {}, {}}), 0);
  std::array<double, 12> storage{};
  EXPECT_EQ(sum_of_strides(matrix_span<stdex::layout_left>(storage.data(), 3, 4)), 4);
}

TEST(MdspanStrideDeathTest, RankZeroHasNoAxisToObserve)
{
  double value = 1.0;
  scalar_span<stdex::layout_left> span(&value);
  EXPECT_DEATH((void)uni20::stride(span, 0), "PRECONDITION");
  test_mdspec<scalar_extents, stdex::layout_right> spec{Region{&value}, {}, {}};
  EXPECT_DEATH((void)spec.stride(0), "PRECONDITION");
}

} // namespace
