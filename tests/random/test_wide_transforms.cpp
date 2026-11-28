#include <uni20/random/fill.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace uni20::random::test
{
namespace
{
#if UNI20_HAS_FLOAT80 || UNI20_HAS_FLOAT128
template <class R> void check_fixed_normals()
{
  constexpr auto words = UnitUniform<R>::word_count;
  using Normal = NormalFor<R>;
  using Circular = ComplexNormalFor<R>;
  static_assert(Uniform<R>::input_type::size == 1 && Uniform<R>::result_type::size == 4 / words);
  static_assert(Normal::input_type::size == words / 2 && Normal::result_type::size == 2);
  static_assert(Circular::input_type::size == words / 2 && Circular::result_type::size == 1);
  R const eps = numeric_limits<R>::epsilon();
  R const grid = eps / 2;
  R const radius = math::sqrt(2 * log_two<R>); // u1 = 1/2
  constexpr int cos_axis[]{1, 0, -1, 0}, sin_axis[]{0, 1, 0, -1};
  for (unsigned axis = 0; axis < 4; ++axis)
    for (int step : {-1, 0, 1})
    {
      SCOPED_TRACE(::testing::Message() << "axis " << axis << ", grid step " << step);
      std::array<std::uint32_t, 2 * words> raw{};
      for (unsigned i = 0; i < words; ++i)
      {
        raw[i] = UINT32_MAX;
        raw[words + i] = step < 0 ? UINT32_MAX : 0;
      }
      raw[words - 1] >>= 1;
      raw.back() += axis << 30;
      if (step > 0) raw[words] = 1u << (words * 32 - numeric_limits<R>::digits());
      typename Normal::input_type input{};
      for (unsigned i = 0; i < raw.size(); ++i)
        input.blocks[i / 4].words[i % 4] = raw[i];

      // At one grid step sin(delta) rounds to delta and cos(delta) rounds to 1.
      R const delta = (2 * step * grid) * pi<R>;
      R const re = radius * (cos_axis[axis] - sin_axis[axis] * delta);
      R const im = radius * (sin_axis[axis] + cos_axis[axis] * delta);
      auto check = [eps](R x, R expected) { EXPECT_TRUE(math::abs(x - expected) <= 8 * eps * math::abs(expected)); };
      auto const pair = Normal{}(input);
      check(pair.values[0], re);
      check(pair.values[1], im);
      auto const z = Circular{}(input).values[0];
      check(z.real(), re * math::sqrt(R{0.5}));
      check(z.imag(), im * math::sqrt(R{0.5}));

      auto const shifted = Normal{R{0.25}, R{1.5}}(input);
      for (unsigned i = 0; i < 2; ++i)
      {
        R const component = i == 0 ? re : im;
        R const expected = R{0.25} + R{1.5} * component;
        EXPECT_TRUE(math::abs(shifted.values[i] - expected) <= 8 * eps * (R{0.25} + R{1.5} * math::abs(component)));
      }
    }
}
#endif

#if UNI20_HAS_FLOAT80
TEST(WideRandomTransformTest, Binary80UsesEveryBitOfEachWordPair)
{
  using U = UnitUniform<float80>;
  constexpr auto ones = UINT32_MAX;
  static_assert(U::closed_open({0, 0}) == 0 && U::open_closed({0, 0}) == 0x1p-64L);
  static_assert(U::closed_open({ones, ones}) == 1 - 0x1p-64L && U::open_closed({ones, ones}) == 1);
  constexpr auto draw = Uniform<float80>{}({{{{1, 0x40000000, 3, 0x80000000}}}});
  static_assert(draw.values[0] == 0.25L + 0x1p-64L && draw.values[1] == 0.5L + 3 * 0x1p-64L);
  check_fixed_normals<float80>();
}
#endif

#if UNI20_HAS_FLOAT128
TEST(WideRandomTransformTest, Binary128TakesTheHigh113BitsOfOneBlock)
{
  using U = UnitUniform<float128>;
  constexpr auto ones = UINT32_MAX;
  constexpr float128 grid = 0x1p-113;
  static_assert(U::closed_open({0x7fff, 0, 0, 0}) == 0 && U::open_closed({0, 0, 0, 0}) == grid);
  static_assert(U::closed_open({ones, ones, ones, ones}) == 1 - grid && U::open_closed({ones, ones, ones, ones}) == 1);
  constexpr auto mixed = U::closed_open({0xffff, 1, 2, 0x40000000});
  static_assert(mixed == float128{0.25} + 0x1p-63 + 0x1p-96 + grid);
  check_fixed_normals<float128>();
}
#endif

#if UNI20_HAS_MPREAL
// Independent decoding: concatenate individual bits as a binary fraction, in draw order.
mpreal reference_unit(Identity leaf, unsigned position, int bits, bool upper, Precision reference)
{
  auto const outer = value_block(leaf, position);
  Identity nested{(std::uint64_t{outer.words[1]} << 32) + outer.words[0],
                  (std::uint64_t{outer.words[3]} << 32) + outer.words[2]};
  mpreal result(0, reference), place(1, reference);
  mpreal const half(0.5, reference);
  for (int b = 0; b < bits; ++b)
  {
    place *= half;
    auto block = value_block(nested, b / 128);
    if ((block.words[b % 128 / 32] >> (31 - b % 32)) & 1) result += place;
  }
  if (upper) result += place;
  return result;
}

TEST(WideRandomTransformTest, NestedDrawKeepsBothUniformPrefixesAndOtherElementPositions)
{
  using R = mpreal;
  using U = UnitUniform<R>;
  using Pair = NormalFor<R>;
  using Circular = ComplexNormalFor<R>;
  static_assert(Uniform<R>::input_type::size == 1 && Uniform<R>::result_type::size == 1);
  static_assert(Pair::input_type::size == 2 && Pair::result_type::size == 2);
  static_assert(Circular::input_type::size == 2 && Circular::result_type::size == 1);
  auto const leaf = root(987);
  static_assert(!std::default_initializable<U>);
  EXPECT_THROW((void)U(Precision::exact()), std::logic_error);
  for (unsigned group : {0u, 3u})
  {
    auto const radial_block = value_block(leaf, 2 * group);
    auto const angular_block = value_block(leaf, 2 * group + 1);
    U::words_type radial{}, angular{};
    std::ranges::copy(radial_block.words, radial.begin());
    std::ranges::copy(angular_block.words, angular.begin());
    for (int bits : {1, 31, 32, 128, 129, 256, 2048, 2049})
    {
      SCOPED_TRACE(::testing::Message() << "group " << group << ", bits " << bits);
      auto const p = Precision::bits(bits);
      auto const reference = Precision::bits(bits + 256);
      U const converter{p};
      auto const u1 = converter.open_closed(radial), u2 = converter.closed_open(angular);
      EXPECT_EQ(u1.precision(), p);
      EXPECT_EQ(u2.precision(), p);
      auto const r1 = reference_unit(leaf, 2 * group, bits, true, reference);
      auto const r2 = reference_unit(leaf, 2 * group + 1, bits, false, reference);
      EXPECT_EQ(u1.at(reference), r1);
      EXPECT_EQ(value_at(leaf, 2 * group + 1, Uniform<R>{.converter = converter}).at(reference), r2);
      if (bits == 256)
      {
        auto const low = Precision::bits(128);
        // Both grids share the raw-bit prefix. The upper endpoint numbers the bins from one.
        R const one_bin = math::ldexp(R{1, reference}, -128);
        R const low1 = U{low}.open_closed(radial).at(reference);
        R const low2 = U{low}.closed_open(angular).at(reference);
        EXPECT_TRUE(low1 - one_bin < r1 && r1 <= low1);
        EXPECT_TRUE(low2 <= r2 && r2 < low2 + one_bin);
      }

      if (bits < 32) continue;
      auto const radius = math::sqrt(-2 * math::log(r1));
      auto const angle = (2 * pi<R>.at(reference)) * r2;
      R const expected[] = {radius * math::cos(angle), radius * math::sin(angle)};
      Pair const normal{.converter = converter};
      auto const z = value_at(leaf, group, Circular{.converter = converter});
      auto const scale = math::sqrt(R{0.5, reference});
      auto const eps = numeric_limits<R>::epsilon(p).at(reference);
      for (unsigned i = 0; i < 2; ++i)
      {
        auto const x = value_at(leaf, 2 * group + i, normal);
        auto const part = i == 0 ? z.real() : z.imag();
        EXPECT_EQ(x.precision(), p);
        EXPECT_EQ(part.precision(), p);
        EXPECT_TRUE(math::abs(x.at(reference) - expected[i]) <= 8 * eps * math::abs(expected[i]));
        EXPECT_TRUE(math::abs(part.at(reference) - scale * expected[i]) <= 8 * eps * scale * math::abs(expected[i]));
      }
    }
  }
}

TEST(WideRandomTransformTest, MpfrHostFillKeepsRequestedPrecisionAndOrigin)
{
  auto const p = Precision::bits(193);
  NormalFor<mpreal> const normal{.converter = {p}};
  auto const leaf = root(27);
  Tensor<mpreal, 1> out(5, p);
  random::fill(out, leaf, normal, 3);
  for (unsigned i = 0; i < 5; ++i)
  {
    EXPECT_EQ(out[i].precision(), p);
    EXPECT_EQ(out[i], value_at(leaf, 3 + i, normal));
  }
}
#endif
} // namespace
} // namespace uni20::random::test
