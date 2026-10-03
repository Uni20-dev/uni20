#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>
#include <type_traits>
#include <uni20/core/math.hpp>
#include <uni20/core/mpreal.hpp>
#include <utility>
#include <vector>

using namespace uni20;

static_assert(std::is_nothrow_move_constructible_v<exact_constant>);
static_assert(std::is_nothrow_move_assignable_v<exact_constant>);

TEST(ExactConstant, MoveTransfersValueAndLeavesSourceUnset)
{
  exact_constant source("123456789012345678901234567890/7");
  auto copy = source;
  auto moved = std::move(source);
  EXPECT_FALSE(source.initialized());
  EXPECT_FALSE(source.is_exact());
  EXPECT_FALSE(is_exact(source));
  EXPECT_TRUE(moved.initialized());
  EXPECT_EQ(moved, copy);

  source = exact_constant("2/3");
  EXPECT_EQ(source.to_string(), "2/3");
  EXPECT_EQ(moved, copy);
  source = std::move(moved);
  EXPECT_EQ(source, copy);
  EXPECT_FALSE(moved.initialized());

  // Self-move must not release the value it is about to acquire.
  auto& alias = source;
  source = std::move(alias);
  EXPECT_EQ(source, copy);
  EXPECT_EQ(exact_constant{}, 0);
}

TEST(ExactConstant, CopyAssignmentAndSwapSupportUnsetStates)
{
  exact_constant source("3/7");
  auto value = std::move(source);
  auto empty_copy = source;
  auto empty_move = std::move(source);
  EXPECT_FALSE(empty_copy.initialized());
  EXPECT_FALSE(empty_move.initialized());

  exact_constant destination{5};
  destination = source;
  EXPECT_FALSE(destination.initialized());
  destination = value;
  EXPECT_EQ(destination.to_string(), "3/7");
  EXPECT_EQ(value, destination);
  source.swap(destination);
  EXPECT_EQ(source, value);
  EXPECT_FALSE(destination.initialized());
  std::swap(source, destination);
  EXPECT_FALSE(source.initialized());
  EXPECT_EQ(destination, value);
  empty_copy.swap(source);
  EXPECT_FALSE(empty_copy.initialized());
  EXPECT_FALSE(source.initialized());
  destination = std::move(empty_move);
  EXPECT_FALSE(destination.initialized());

  std::vector<exact_constant> values;
  values.push_back(std::move(source));
  values.push_back(value);
  values.reserve(32);
  EXPECT_FALSE(values[0].initialized());
  EXPECT_EQ(values[1], value);
}

TEST(ExactConstant, UnsetValuesHaveNoNumericalMeaning)
{
  exact_constant unset{1};
  auto value = std::move(unset);
  EXPECT_THROW(unset.native_handle(), std::logic_error);
  EXPECT_THROW(unset.to_string(), std::logic_error);
  EXPECT_THROW(+unset, std::logic_error);
  EXPECT_THROW(-unset, std::logic_error);
  EXPECT_THROW(unset + value, std::logic_error);
  EXPECT_THROW(value + unset, std::logic_error);
  EXPECT_THROW(unset - value, std::logic_error);
  EXPECT_THROW(value - unset, std::logic_error);
  EXPECT_THROW(unset * value, std::logic_error);
  EXPECT_THROW(value * unset, std::logic_error);
  EXPECT_THROW(unset / value, std::logic_error);
  EXPECT_THROW(value / unset, std::logic_error);
  EXPECT_THROW((void)(unset == value), std::logic_error);
  EXPECT_THROW((void)(unset < value), std::logic_error);
  EXPECT_THROW(sqrt(unset), std::logic_error);
  EXPECT_THROW(pow(unset, exact_constant{0}), std::logic_error);
  EXPECT_THROW(pow(value, unset), std::logic_error);
  EXPECT_THROW((void)mpreal{unset}, std::logic_error);
  EXPECT_THROW((void)mpreal{std::move(unset)}, std::logic_error);
  EXPECT_THROW(unset.at(Precision::exact()), std::logic_error);
  EXPECT_THROW(unset.at(Precision::bits(128)), std::logic_error);
}

namespace
{
// Used only in a freshly executed death-test child. Delegate to the original
// allocators so any allocations made before installing these hooks remain valid.
struct GmpAllocationProbe
{
    static inline std::size_t allocations = 0;
    static inline void* (*allocate)(std::size_t) = nullptr;
    static inline void* (*reallocate)(void*, std::size_t, std::size_t) = nullptr;
    static inline void (*release)(void*, std::size_t) = nullptr;

    static void install()
    {
      mp_get_memory_functions(&allocate, &reallocate, &release);
      mp_set_memory_functions(
          [](std::size_t size) -> void* {
            ++allocations;
            return allocate(size);
          },
          [](void* data, std::size_t old_size, std::size_t new_size) -> void* {
            ++allocations;
            return reallocate(data, old_size, new_size);
          },
          release);
    }
};
} // namespace

TEST(ExactConstantDeathTest, MovesDoNotAllocate)
{
  // Never change GMP's process-global hooks in the main test process, where
  // other tests may have started scheduler workers or retained GMP objects.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        GmpAllocationProbe::install();
        bool valid = false;
        std::size_t move_allocations = 0;
        {
          exact_constant source("123456789012345678901234567891/7");
          exact_constant target{9};
          mpreal real(source);
          std::vector<exact_constant> values(3, source);
          auto before = GmpAllocationProbe::allocations;
          auto moved = std::move(source);
          auto empty = std::move(source);
          auto empty_copy = source;
          target = std::move(moved);
          std::swap(target, source);
          auto moved_real = std::move(real);
          values.reserve(32);
          move_allocations = GmpAllocationProbe::allocations - before;
          valid = before > 0 && source.to_string() == "123456789012345678901234567891/7" &&
                  moved_real.exact_value() == source && values[0] == source && !target.initialized() &&
                  !moved.initialized() && !empty.initialized() && !empty_copy.initialized() && !real.initialized();
        } // Exercise destruction of both owning and unset objects before exit.
        std::fprintf(stderr, "move allocations: %zu\n", move_allocations);
        std::_Exit(valid && move_allocations == 0 ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "move allocations: 0");
}
