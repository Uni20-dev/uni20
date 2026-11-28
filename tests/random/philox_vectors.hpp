#pragma once

#include <uni20/random/philox.hpp>

namespace uni20::random::test
{

struct PhiloxVector
{
    Philox4x32Block counter;
    Philox4x32Key key;
    Philox4x32Block expected;
};

// All three philox4x32 10 cases from Random123 tests/kat_vectors:
// https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors
inline constexpr PhiloxVector known_answers[] = {
    {{{0, 0, 0, 0}}, {{0, 0}}, {{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}}},
    {{{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}},
     {{0xffffffff, 0xffffffff}},
     {{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}}},
    {{{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}},
     {{0xa4093822, 0x299f31d0}},
     {{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}}},
};

} // namespace uni20::random::test
