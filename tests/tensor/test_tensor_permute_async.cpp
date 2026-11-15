#include <uni20/tensor/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

namespace
{
using matrix = uni20::DenseMatrix<double>;
using async_matrix = uni20::async::Async<matrix>;
using alias_type = decltype(uni20::async::permute<1, 0>(std::declval<async_matrix&>()));
using const_alias_type = decltype(uni20::async::permute<1, 0>(std::declval<async_matrix const&>()));
static_assert(uni20::MutableTensorView<uni20::async::async_value_t<alias_type>>);
static_assert(!uni20::MutableTensorView<uni20::async::async_value_t<const_alias_type>>);
static_assert(uni20::async::is_async_alias_v<uni20::async::async_value_t<alias_type>>);

matrix make_matrix()
{
  matrix result(2, 3);
  result[1, 2] = 12;
  return result;
}

struct ParentFailure
{};
} // namespace

TEST(TensorPermuteAsyncTest, SharesQueueAndWritesThrough)
{
  async_matrix parent(make_matrix());
  auto alias = uni20::async::permute<1, 0>(parent);
  EXPECT_EQ(&alias.queue(), &parent.queue());
  EXPECT_EQ(parent.storage().use_count(), 2);
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix replacement(3, 2);
  replacement[2, 1] = 47;
  uni20::async::Async<matrix> source(std::move(replacement));
  alias = source;
  scheduler.run_all();
  EXPECT_EQ((parent.get_wait().mdspan()[1, 2]), 47);
}

TEST(TensorPermuteAsyncTest, PendingParentIsNotResolvedDuringAliasCreation)
{
  async_matrix parent;
  auto alias = uni20::async::permute<1, 0>(parent);
  uni20::async::DebugScheduler scheduler;
  scheduler.schedule([](uni20::async::WriteBuffer<matrix> writer) static -> uni20::async::AsyncTask {
    (co_await writer).emplace(make_matrix());
    co_return;
  }(parent.write()));
  auto const& result = alias.get_wait(scheduler);
  EXPECT_EQ(result.extent(0), 3);
  EXPECT_EQ(result.extent(1), 2);
  EXPECT_EQ((result.mdspan()[2, 1]), 12);
}

TEST(TensorPermuteAsyncTest, NestedAliasesOutliveParentAndIntermediateHandles)
{
  auto alias = [] {
    async_matrix parent(make_matrix());
    auto first = uni20::async::permute<1, 0>(parent);
    auto second = uni20::async::permute<1, 0>(first);
    EXPECT_EQ(&second.queue(), &parent.queue());
    return second;
  }();
  EXPECT_EQ((alias.get_wait().mdspan()[1, 2]), 12);
}

TEST(TensorPermuteAsyncTest, PropagatesParentFailure)
{
  async_matrix parent;
  auto alias = uni20::async::permute<1, 0>(std::as_const(parent));
  uni20::async::DebugScheduler scheduler;
  scheduler.schedule([](uni20::async::WriteBuffer<matrix> writer) static -> uni20::async::AsyncTask {
    static_cast<void>(writer);
    throw ParentFailure{};
    co_return;
  }(parent.write()));
  EXPECT_THROW(static_cast<void>(alias.get_wait(scheduler)), ParentFailure);
}

#if UNI20_HAS_MPREAL
TEST(TensorPermuteAsyncTest, ReadsPrecisionFromTheParentEpoch)
{
  using mp_matrix = uni20::DenseMatrix<uni20::mpreal>;
  auto p = uni20::Precision::bits(192), q = uni20::Precision::bits(80);
  uni20::async::Async<mp_matrix> parent;
  auto alias = uni20::async::permute<1, 0>(parent);
  uni20::async::DebugScheduler scheduler;
  scheduler.schedule([](uni20::async::WriteBuffer<mp_matrix> writer, uni20::Precision p,
                        uni20::Precision q) static -> uni20::async::AsyncTask {
    auto& tensor = (co_await writer).emplace(2, 3, p);
    tensor[1, 2] = uni20::mpreal(11, p);
    tensor.default_precision(q);
    co_return;
  }(parent.write(), p, q));
  auto const& result = alias.get_wait(scheduler);
  EXPECT_EQ(result.default_precision(), q);
  EXPECT_EQ((result.mdspan()[2, 1].precision()), p);
}
#endif
