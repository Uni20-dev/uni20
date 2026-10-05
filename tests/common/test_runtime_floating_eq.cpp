// Exercise the debug forms even in Release; NDEBUG no-ops are tested separately.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../numerics/precision_cases.hpp"
#include <uni20/common/trace.hpp>

#include <csignal>
#include <cstdlib>

namespace
{
using namespace uni20::test;

using RuntimeCases =
    testing::Types<PrecisionContext<float, false>, PrecisionContext<double, false>, PrecisionContext<double, true>
#if UNI20_HAS_FLOAT80
                   ,
                   PrecisionContext<uni20::float80, false>
#endif
#if UNI20_HAS_FLOAT128
                   ,
                   PrecisionContext<uni20::float128, false>
#endif
#if UNI20_ENABLE_MPFR
                   ,
                   PrecisionContext<uni20::mpreal, false, 128>, PrecisionContext<uni20::mpreal, false, 256>
#endif
                   >;

template <class C> struct Evaluations
{
    bool failing;
    int left_count = 0, right_count = 0, tolerance_count = 0, context_count = 0;

    void count(int& calls)
    {
      ++calls;
      // Distinguish repeated evaluation from the assertion's expected SIGABRT.
      if (failing && calls != 1) std::_Exit(81);
    }

    auto left()
    {
      this->count(left_count);
      return C::scalar(1);
    }
    auto right()
    {
      this->count(right_count);
      return C::scalar(failing ? 2 : 1);
    }
    int tolerance()
    {
      this->count(tolerance_count);
      return 0;
    }
    std::string context()
    {
      this->count(context_count);
      return "evaluation-context";
    }
};

template <class C> void check_evaluations(auto invoke, bool with_extras = true)
{
  Evaluations<C> passing{false};
  invoke(passing);
  EXPECT_EQ(passing.left_count, 1);
  EXPECT_EQ(passing.right_count, 1);
  EXPECT_EQ(passing.tolerance_count, with_extras ? 1 : 0);
  EXPECT_EQ(passing.context_count, with_extras ? 1 : 0);

  GTEST_FLAG_SET(death_test_style, "fast");
  EXPECT_EXIT(
      {
        Evaluations<C> failing{true};
        invoke(failing);
        std::_Exit(82);
      },
      testing::KilledBySignal(SIGABRT), with_extras ? "evaluation-context" : "FLOATING_EQ");
}

template <class C> class RuntimeFloatingEqDeathTest : public testing::Test {};
TYPED_TEST_SUITE(RuntimeFloatingEqDeathTest, RuntimeCases);

TYPED_TEST(RuntimeFloatingEqDeathTest, CheckEvaluatesOnce)
{
  check_evaluations<TypeParam>([](auto& c) { CHECK_FLOATING_EQ(c.left(), c.right(), c.tolerance(), c.context()); });
}

TYPED_TEST(RuntimeFloatingEqDeathTest, PreconditionEvaluatesOnce)
{
  check_evaluations<TypeParam>(
      [](auto& c) { PRECONDITION_FLOATING_EQ(c.left(), c.right(), c.tolerance(), c.context()); });
}

TYPED_TEST(RuntimeFloatingEqDeathTest, DebugCheckEvaluatesOnce)
{
  check_evaluations<TypeParam>(
      [](auto& c) { DEBUG_CHECK_FLOATING_EQ(c.left(), c.right(), c.tolerance(), c.context()); });
}

TYPED_TEST(RuntimeFloatingEqDeathTest, DebugPreconditionEvaluatesOnce)
{
  check_evaluations<TypeParam>(
      [](auto& c) { DEBUG_PRECONDITION_FLOATING_EQ(c.left(), c.right(), c.tolerance(), c.context()); });
}

TYPED_TEST(RuntimeFloatingEqDeathTest, DefaultToleranceEvaluatesOnce)
{
  check_evaluations<TypeParam>([](auto& c) { CHECK_FLOATING_EQ(c.left(), c.right()); }, false);
  check_evaluations<TypeParam>([](auto& c) { PRECONDITION_FLOATING_EQ(c.left(), c.right()); }, false);
  check_evaluations<TypeParam>([](auto& c) { DEBUG_CHECK_FLOATING_EQ(c.left(), c.right()); }, false);
  check_evaluations<TypeParam>([](auto& c) { DEBUG_PRECONDITION_FLOATING_EQ(c.left(), c.right()); }, false);
}

TEST(RuntimeFloatingEqDeathTest, ContextWithoutExplicitTolerance)
{
  using C = PrecisionContext<double, false>;
  GTEST_FLAG_SET(death_test_style, "fast");
  EXPECT_EXIT(
      {
        Evaluations<C> failing{true};
        CHECK_FLOATING_EQ(failing.left(), failing.right(), failing.context());
      },
      testing::KilledBySignal(SIGABRT), "evaluation-context");
}
} // namespace
