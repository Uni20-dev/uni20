#include <uni20/common/half_int.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace
{
using Small = uni20::basic_half_int<std::int16_t>;
using Large = uni20::basic_half_int<std::int64_t>;

static_assert(std::is_convertible_v<Small, Large>);
static_assert(std::is_nothrow_constructible_v<Large, Small>);
static_assert(std::is_assignable_v<Large&, Small>);
static_assert(!std::is_convertible_v<Large, Small>);
static_assert(std::is_constructible_v<Small, Large>);
static_assert(!std::is_nothrow_constructible_v<Small, Large>);
static_assert(!std::is_assignable_v<Small&, Large>);
static_assert(std::is_trivially_copyable_v<Large>);

constexpr uni20::half_int half = uni20::from_twice(1);
static_assert(half.twice() == 1);
static_assert(Small{uni20::from_twice(std::int64_t{-3})}.twice() == -3);
static_assert([] {
  Large value;
  value = uni20::from_twice(std::int16_t{-1});
  return value.twice() == -1;
}());

template <typename T> class HalfIntConversion : public ::testing::Test {};
using NarrowStorage = ::testing::Types<std::int8_t, std::int16_t, std::int32_t>;
TYPED_TEST_SUITE(HalfIntConversion, NarrowStorage);

template <typename T> class HalfIntParsing : public ::testing::Test {};
using ParserStorage = ::testing::Types<std::int8_t, std::int64_t>;
TYPED_TEST_SUITE(HalfIntParsing, ParserStorage);

TYPED_TEST(HalfIntParsing, DecimalBoundariesPreserveEndpointsAndRejectUnderflow)
{
  using T = TypeParam;
  using Half = uni20::basic_half_int<T>;
  auto const minimum = std::numeric_limits<T>::min();
  auto const maximum = std::numeric_limits<T>::max();
  auto const lower = std::to_string(minimum / 2);
  auto const upper = std::to_string(maximum / 2);
  EXPECT_EQ(Half::parse(lower).twice(), minimum);
  EXPECT_EQ(Half::parse(lower + ".0").twice(), minimum);
  EXPECT_EQ(Half::parse(upper + ".5").twice(), maximum);
  EXPECT_EQ(Half::parse(std::to_string(minimum / 2 + 1) + ".5").twice(), minimum + 1);
  EXPECT_THROW(static_cast<void>(Half::parse(lower + ".5")), std::overflow_error);
}

TYPED_TEST(HalfIntConversion, WideningAndCheckedNarrowingPreserveDoubledValues)
{
  using T = TypeParam;
  using Half = uni20::basic_half_int<T>;
  for (T value : {std::numeric_limits<T>::min(), T{-3}, T{-1}, T{0}, T{1}, T{3}, std::numeric_limits<T>::max()})
  {
    Half const small = uni20::from_twice(value);
    Large const large = small;
    EXPECT_EQ(large.twice(), value);
    EXPECT_EQ(Half{large}.twice(), value);
    Large assigned;
    assigned = small;
    EXPECT_EQ(assigned.twice(), value);
  }
}

TYPED_TEST(HalfIntConversion, NarrowingRejectsOutOfRangeDoubledValues)
{
  using T = TypeParam;
  using Half = uni20::basic_half_int<T>;
  auto const below = uni20::from_twice(std::int64_t{std::numeric_limits<T>::min()} - 1);
  auto const above = uni20::from_twice(std::int64_t{std::numeric_limits<T>::max()} + 1);
  EXPECT_THROW(static_cast<void>(Half{below}), std::overflow_error);
  EXPECT_THROW(static_cast<void>(Half{above}), std::overflow_error);
  EXPECT_THROW(static_cast<void>(Half{uni20::from_twice(std::numeric_limits<std::int64_t>::min())}),
               std::overflow_error);
  EXPECT_THROW(static_cast<void>(Half{uni20::from_twice(std::numeric_limits<std::int64_t>::max())}),
               std::overflow_error);
}

TEST(HalfIntConversion, EqualWidthDifferentStorageTypesAreImplicit)
{
  // long and long long have the same range on LP64; elsewhere test widening.
  using From = uni20::basic_half_int<long>;
  using To = uni20::basic_half_int<long long>;
  static_assert(std::is_convertible_v<From, To>);
  static_assert(std::is_nothrow_constructible_v<To, From>);
  for (long value : {std::numeric_limits<long>::min(), -1L, 1L, std::numeric_limits<long>::max()})
  {
    To const result = uni20::from_twice(value);
    EXPECT_EQ(result.twice(), value);
  }
}
} // namespace
