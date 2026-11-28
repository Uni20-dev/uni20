#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <type_traits>
#include <uni20/random/stream.hpp>

namespace uni20::random::test
{
static_assert(std::uniform_random_bit_generator<PhiloxStream>);
static_assert(std::same_as<PhiloxStream::result_type, std::uint32_t>);
static_assert(std::is_trivially_copyable_v<PhiloxStream>);

TEST(RandomIdentityTest, DerivationAndValuesUseDisjointDomains)
{
  auto const node = root(42);
  EXPECT_EQ(node, root(42));
  EXPECT_NE(node, root(43));
  constexpr std::uint64_t limit = UINT64_C(1) << 63;
  for (std::uint64_t p : std::array<std::uint64_t, 4>{0, 1, UINT64_C(1) << 32, limit - 1})
  {
    auto id = child(node, p);
    EXPECT_EQ(id, child(node, p));
    EXPECT_NE(id, child(node, p ^ 1));
    auto const w = value_block(node, p);
    Identity from_value{(std::uint64_t{w.words[1]} << 32) | w.words[0], (std::uint64_t{w.words[3]} << 32) | w.words[2]};
    EXPECT_NE(id, from_value);
  }
  EXPECT_THROW((void)child(node, limit), std::out_of_range);
  EXPECT_THROW((void)value_block(node, limit), std::out_of_range);
}

TEST(RandomStreamTest, WordOrderDiscardCopyAndShuffle)
{
  auto const leaf = child(root(35), 0);
  for (unsigned used = 0; used < 5; ++used)
    for (auto skip : std::array<std::uint64_t, 8>{0, 1, 2, 3, 4, 5, 39, UINT64_C(1) << 45})
    {
      PhiloxStream stream(leaf);
      for (unsigned i = 0; i < used; ++i)
        (void)stream();
      stream.discard(skip);
      auto copy = stream;
      for (auto i = skip + used; i < skip + used + 9; ++i)
      {
        auto w = value_block(leaf, i / 4).words[i % 4];
        EXPECT_EQ(stream(), w);
        EXPECT_EQ(copy(), w);
      }
    }
  std::array<int, 32> a{}, b{};
  std::iota(a.begin(), a.end(), 0);
  b = a;
  PhiloxStream stream(leaf);
  std::shuffle(a.begin(), a.end(), stream);
  EXPECT_NE(a, b);
  std::ranges::sort(a);
  EXPECT_EQ(a, b);
}

TEST(RandomStreamTest, NeverWrapsAtTheLastWord)
{
  PhiloxStream stream(root(13));
  stream.discard(UINT64_MAX);
  stream.discard(UINT64_MAX);
  EXPECT_THROW(stream.discard(3), std::out_of_range);
  auto const last = value_block(root(13), (UINT64_C(1) << 63) - 1);
  EXPECT_EQ(stream(), last.words[2]);
  EXPECT_EQ(stream(), last.words[3]);
  stream.discard(0);
  EXPECT_THROW((void)stream(), std::out_of_range);
  EXPECT_THROW(stream.discard(1), std::out_of_range);
}

#if defined(__cpp_lib_philox_engine)
TEST(RandomStreamTest, MatchesStandardEngineAcrossCounterCarry)
{
  std::seed_seq seed{31, 73};
  std::uint32_t key[2];
  seed.generate(key, key + 2);
  Identity const leaf{(std::uint64_t{key[1]} << 32) | key[0], UINT64_C(0x123456789abcdef0)};
  std::philox4x32 standard(seed);
  standard.set_counter({0x12345678, 0x9abcdef0, 0x80000000, 0xffffffff});
  PhiloxStream stream(leaf);
  stream.discard(UINT64_C(0xffffffff) * 4);
  for (int i = 0; i < 12; ++i)
    EXPECT_EQ(stream(), standard());
}
#endif
} // namespace uni20::random::test
