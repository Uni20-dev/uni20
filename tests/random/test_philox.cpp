#include "philox_vectors.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>

namespace uni20::random::test
{

static_assert(std::ranges::equal(philox4x32_10({}, {}).words, known_answers[0].expected.words));

TEST(Philox4x32Test, MatchesRandom123KnownAnswers)
{
  for (auto const& v : known_answers)
  {
    auto const block = philox4x32_10(v.counter, v.key);
    for (int lane = 0; lane < 4; ++lane)
      EXPECT_EQ(block.words[lane], v.expected.words[lane]) << "lane " << lane;
  }
}

#if defined(__cpp_lib_philox_engine)
TEST(Philox4x32Test, MatchesStandardEngine)
{
  std::seed_seq seed{123, 456};
  Philox4x32Key key;
  seed.generate(key.words, key.words + 2);
  std::philox4x32 engine(seed);

  auto counter = known_answers[2].counter;
  // set_counter takes the most significant word first. The engine emits the
  // four lanes in order, then advances the least significant word (word 0).
  engine.set_counter({counter.words[3], counter.words[2], counter.words[1], counter.words[0]});
  for (int block = 0; block < 3; ++block, ++counter.words[0])
    for (auto word : philox4x32_10(counter, key).words)
      EXPECT_EQ(word, engine());
}
#endif

} // namespace uni20::random::test
