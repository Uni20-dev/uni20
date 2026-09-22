#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <functional>
#include <utility>

namespace
{
struct RecordingStorage : uni20::HostStorage
{
    static inline int zero_allocations = 0;
    static inline int uninitialized_allocations = 0;

    static void reset_counts()
    {
      zero_allocations = 0;
      uninitialized_allocations = 0;
    }

    template <class T>
    static auto make_storage(std::size_t size, uni20::StorageInitialization initialization) -> storage_t<T>
    {
      if (initialization == uni20::StorageInitialization::Zero)
        ++zero_allocations;
      else
        ++uninitialized_allocations;
      return HostStorage::make_storage<T>(size, initialization);
    }
};

using vector_type = uni20::Tensor<double, 1, RecordingStorage>;
using matrix_type = uni20::Tensor<double, 2, RecordingStorage>;
using scalar_type = uni20::Tensor<double, 0, RecordingStorage>;

vector_type make_vector()
{
  vector_type value(2);
  value[0] = 3.0;
  value[1] = 4.0;
  return value;
}
} // namespace

TEST(AsyncInitializationTest, TransformAllocatesForOverwriteAndReusesConstructedOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<vector_type> input = make_vector();
  uni20::async::Async<vector_type> output;
  RecordingStorage::reset_counts();

  uni20::assign_transform(output, std::negate<>{}, input);
  auto const& first = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ(first[0], -3.0);
  EXPECT_DOUBLE_EQ(first[1], -4.0);
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 1);
  auto const* const allocation = first.data();

  uni20::assign_transform(output, [](double value) { return 2.0 * value; }, input);
  auto const& second = output.get_wait(scheduler);
  EXPECT_EQ(second.data(), allocation);
  EXPECT_DOUBLE_EQ(second[0], 6.0);
  EXPECT_DOUBLE_EQ(second[1], 8.0);
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 1);
}

TEST(AsyncInitializationTest, ExplicitScalarReductionsAllocateForOverwrite)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<vector_type> input = make_vector();
  matrix_type matrix_value(2, 2);
  matrix_value[0, 0] = 1.0;
  matrix_value[0, 1] = -2.0;
  matrix_value[1, 0] = 3.0;
  matrix_value[1, 1] = -4.0;
  uni20::async::Async<matrix_type> matrix = std::move(matrix_value);
  uni20::async::Async<scalar_type> norm;
  uni20::async::Async<scalar_type> inner;
  uni20::async::Async<scalar_type> matrix_norm;
  RecordingStorage::reset_counts();

  uni20::norm(norm, input);
  uni20::inner_product(inner, input, input);
  uni20::linalg::matrix_norm(matrix_norm, matrix, uni20::linalg::MatrixNorm::Infinity);
  EXPECT_DOUBLE_EQ(norm.get_wait(scheduler)[], 5.0);
  EXPECT_DOUBLE_EQ(inner.get_wait(scheduler)[], 25.0);
  EXPECT_DOUBLE_EQ(matrix_norm.get_wait(scheduler)[], 7.0);
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 3);

  auto const* const allocation = norm.get_wait(scheduler).data();
  uni20::inner_product(norm, input, input);
  EXPECT_EQ(norm.get_wait(scheduler).data(), allocation);
  EXPECT_DOUBLE_EQ(norm.get_wait(scheduler)[], 25.0);
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 3);
}

TEST(AsyncInitializationTest, MaterializationCoroutineAllocatesForOverwrite)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<vector_type> input = make_vector();
  uni20::async::Async<vector_type> output;
  RecordingStorage::reset_counts();

  // The public make_tensor result fixes HostStorage; inject the recording policy
  // into its coroutine to observe the actual materialization allocation.
  uni20::async::schedule(
      uni20::detail::co_make_tensor(uni20::linalg::CpuReferenceBackend{}, output.write(), input.read()));
  auto const& result = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ(result[0], 3.0);
  EXPECT_DOUBLE_EQ(result[1], 4.0);
  EXPECT_NE(result.data(), input.get_wait(scheduler).data());
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 1);
}

TEST(AsyncInitializationTest, DeferredOwnersRetainOrdinaryConstructorFallback)
{
  using deferred_vector = uni20::test::DeferredHostTensor<double, 1>;
  using deferred_scalar = uni20::test::DeferredHostTensor<double, 0>;
  static_assert(!std::constructible_from<deferred_vector, uni20::uninitialized_t, deferred_vector::extents_type>);
  static_assert(!std::constructible_from<deferred_scalar, uni20::uninitialized_t>);
  deferred_vector value(2);
  {
    auto lease = uni20::test::acquire_host_write_access_sync(value);
    lease.mdspan()[0] = 3.0;
    lease.mdspan()[1] = 4.0;
  }

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<deferred_vector> input = std::move(value);
  uni20::async::Async<deferred_vector> output;
  uni20::async::Async<deferred_scalar> norm;
  uni20::assign_transform(output, std::negate<>{}, input);
  uni20::norm(norm, output);

  auto output_lease = uni20::test::acquire_host_read_access_sync(output.get_wait(scheduler));
  EXPECT_DOUBLE_EQ(output_lease.mdspan()[0], -3.0);
  EXPECT_DOUBLE_EQ(output_lease.mdspan()[1], -4.0);
  auto norm_lease = uni20::test::acquire_host_read_access_sync(norm.get_wait(scheduler));
  EXPECT_DOUBLE_EQ(norm_lease.mdspan()[], 5.0);
}
