#include <array>
#include <gtest/gtest.h>
#include <uni20/mdspan/layout_signed_stride.hpp>
#include <uni20/random/fill.hpp>
#include <uni20/tensor/tensor.hpp>

namespace uni20::random::test
{
namespace
{
struct TwoBlockTransform
{
    using input_type = BlockGroup<2>;
    using result_type = ValueBlock<double, 2>;
    result_type operator()(input_type const& group) const
    {
      return {{static_cast<double>(group.blocks[0].words[3]), static_cast<double>(group.blocks[1].words[1])}};
    }
};

TEST(DenseRandomFillTest, MultiBlockGroupUsesBothConsecutivePositions)
{
  Tensor<double, 1> out(11);
  auto const leaf = root(39);
  random::fill(out, leaf, TwoBlockTransform{}, 3);
  for (unsigned i = 0; i < 11; ++i)
  {
    unsigned const offset = i + 3;
    auto const expected = value_block(leaf, offset).words[offset % 2 ? 1 : 3];
    EXPECT_EQ(out[i], expected);
    EXPECT_EQ(value_at(leaf, offset, TwoBlockTransform{}), expected);
  }
  EXPECT_THROW((void)value_group(leaf, UINT64_C(1) << 62, TwoBlockTransform{}), std::out_of_range);
}
} // namespace

TEST(DenseRandomFillTest, ReversedPaddedStorageUsesSpanStart)
{
  std::array<double, 15> storage{};
  using E = dextents<index_type, 2>;
  layout_signed_stride::mapping<E> mapping{E{2, 3}, std::array<index_type, 2>{-7, 2}};
  auto view = mdspan{storage.data() + 1, mapping};
  auto leaf = child(root(42), 11);
  for (auto origin : {0u, 31u})
  {
    random::fill(view, leaf, Uniform<double>{}, origin);
    for (unsigned i = 0; i < storage.size(); ++i)
    {
      bool touched = i == 1 || i == 3 || i == 5 || i == 8 || i == 10 || i == 12;
      EXPECT_EQ(storage[i], touched ? value_at(leaf, origin + i - 1, Uniform<double>{}) : 0);
    }
  }
}

TEST(DenseRandomFillTest, HostTensorAndUnalignedSubfillsMatchGroupedTransform)
{
  auto leaf = child(root(57), 2);
  Tensor<float, 1> all(19);
  random::fill(all, leaf, Uniform<float>{});
  std::array<float, 19> split{};
  for (unsigned start : {0u, 1u, 6u, 13u})
  {
    unsigned end = start == 0 ? 1 : start == 1 ? 6 : start == 6 ? 13 : 19;
    auto sub = mdspan{split.data() + start, dextents<index_type, 1>{end - start}};
    random::fill(sub, leaf, Uniform<float>{}, start);
  }
  for (unsigned i = 0; i < split.size(); ++i)
  {
    auto block = Uniform<float>{}({value_block(leaf, i / 4)});
    EXPECT_EQ(all[i], block.values[i % 4]);
    EXPECT_EQ(split[i], all[i]);
  }
}

TEST(DenseRandomFillTest, EmptyRankZeroAndAddressLimits)
{
  auto leaf = root(27);
  complex<double> scalar{};
  auto single = mdspan{&scalar, extents<index_type>{}};
  constexpr std::uint64_t limit = UINT64_C(1) << 63;
  random::fill(single, leaf, ComplexNormal{}, limit - 1);
  EXPECT_EQ(scalar, value_at(leaf, limit - 1, ComplexNormal{}));
  EXPECT_THROW(random::fill(single, leaf, ComplexNormal{}, limit), std::out_of_range);
  double storage[2]{};
  auto empty = mdspan{storage, dextents<index_type, 1>{0}};
  random::fill(empty, leaf, Uniform<double>{}, UINT64_MAX);
  auto view = mdspan{storage, dextents<index_type, 1>{2}};
  EXPECT_THROW(random::fill(view, leaf, Uniform<double>{}, UINT64_MAX), std::out_of_range);
  EXPECT_EQ(storage[0], 0.);
}
} // namespace uni20::random::test
