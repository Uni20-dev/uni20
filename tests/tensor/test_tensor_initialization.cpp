#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>

namespace
{
struct RecordingStorage : uni20::HostStorage
{
    static inline int zero_allocations = 0;
    static inline int uninitialized_allocations = 0;

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

template <class T> void expect_zero_construction()
{
  uni20::DenseMatrix<T> matrix(3, 2);
  for (uni20::index_type i = 0; i < 3; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      EXPECT_TRUE((matrix[i, j] == T{0}));
  uni20::ScalarTensor<T> scalar;
  EXPECT_TRUE(scalar[] == T{0});
}
} // namespace

TEST(TensorInitializationTest, ShapeConstructionInitializesRealAndComplexValues)
{
  expect_zero_construction<int>();
  expect_zero_construction<float>();
  expect_zero_construction<double>();
  expect_zero_construction<long double>();
  expect_zero_construction<uni20::complex<double>>();
#if UNI20_HAS_FLOAT128
  expect_zero_construction<uni20::float128>();
  expect_zero_construction<uni20::complex<uni20::float128>>();
#endif
}

TEST(TensorInitializationTest, StridedAndStaticExtentsInitializeOwnedElements)
{
  using tensor_type = uni20::StridedTensor<double, 2>;
  tensor_type matrix(tensor_type::extents_type{2, 3}, std::array<uni20::index_type, 2>{1, 4});
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      EXPECT_EQ((matrix[i, j]), 0.0);

  using static_type = uni20::Tensor<double, 2, uni20::HostStorage, stdex::layout_left, uni20::DefaultAccessorFactory,
                                    stdex::extents<uni20::index_type, 2, 3>>;
  static_type fixed(static_type::extents_type{});
  EXPECT_EQ((fixed[1, 2]), 0.0);
  uni20::DenseMatrix<double> empty(0, 4);
  EXPECT_EQ(empty.storage().size(), 0u);
}

TEST(TensorInitializationTest, ExplicitUninitializedAndOverwritePreparationAvoidZeroAllocation)
{
  using tensor_type = uni20::Tensor<double, 2, RecordingStorage>;
  RecordingStorage::zero_allocations = 0;
  RecordingStorage::uninitialized_allocations = 0;
  tensor_type output(uni20::uninitialized, 2, 3);
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 1);

  uni20::prepare_output(output, tensor_type::extents_type{3, 4});
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 2);
  output[0, 0] = 17.0;
  auto* const allocation = output.data();
  uni20::prepare_output(output, output.extents());
  EXPECT_EQ(output.data(), allocation);
  EXPECT_EQ((output[0, 0]), 17.0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 2);

  output.reset_shape(tensor_type::extents_type{1, 2});
  EXPECT_EQ(RecordingStorage::zero_allocations, 1);
  EXPECT_EQ((output[0, 0]), 0.0);
  EXPECT_EQ((output[0, 1]), 0.0);
}

TEST(TensorInitializationTest, MaterializationAndAsyncOutputAllocateForOverwrite)
{
  using scalar_type = uni20::complex<double>;
  using tensor_type = uni20::Tensor<scalar_type, 2, RecordingStorage>;
  tensor_type source(2, 2);
  source[0, 0] = scalar_type{3.0, 4.0};
  source[1, 1] = 2.0;
  RecordingStorage::zero_allocations = 0;
  RecordingStorage::uninitialized_allocations = 0;
  tensor_type materialized(uni20::conj(source));
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 1);
  EXPECT_EQ((materialized[0, 0]), (scalar_type{3.0, -4.0}));
  EXPECT_EQ((materialized[0, 1]), scalar_type{0.0});
  EXPECT_EQ((materialized[1, 1]), scalar_type{2.0});

  auto storage = uni20::async::make_unconstructed_shared_storage<tensor_type>();
  auto& output = uni20::prepare_output(storage, source.extents());
  EXPECT_EQ(RecordingStorage::zero_allocations, 0);
  EXPECT_EQ(RecordingStorage::uninitialized_allocations, 2);
  EXPECT_EQ(output.extents(), source.extents());
}

TEST(TensorInitializationTest, NontrivialElementsRemainLive)
{
  uni20::Tensor<std::string, 1> values(uni20::uninitialized, 2);
  values[0] = "constructed";
  EXPECT_EQ(values[0], "constructed");
  EXPECT_EQ(values[1], "");
  uni20::HostBuffer<std::string> buffer(1, "preserved");
  buffer.resize(2, uni20::StorageInitialization::Zero);
  EXPECT_EQ(buffer[0], "preserved");
  EXPECT_EQ(buffer[1], "");
}

TEST(TensorInitializationTest, MoveOnlyAndNonassignableElementsCanBeAllocated)
{
  uni20::HostBuffer<std::unique_ptr<int>> pointers(2);
  pointers[0] = std::make_unique<int>(7);
  EXPECT_EQ(*pointers[0], 7);
  EXPECT_EQ(pointers[1], nullptr);
  uni20::Tensor<std::unique_ptr<int>, 1> tensor(2);
  EXPECT_EQ(tensor[0], nullptr);

  struct ConstElement
  {
      int const value;
  };
  static_assert(uni20::uninitialized_ok<ConstElement>);
  uni20::HostBuffer<ConstElement> constants(2, uni20::StorageInitialization::Zero);
  EXPECT_EQ(constants[0].value, 0);
  EXPECT_EQ(constants[1].value, 0);
  constants.resize(3, uni20::StorageInitialization::Zero);
  EXPECT_EQ(constants[2].value, 0);

  struct Nonassignable
  {
      ~Nonassignable() {}
      Nonassignable& operator=(Nonassignable const&) = delete;
      int value;
  };
  uni20::Tensor<Nonassignable, 1> objects(2);
  EXPECT_EQ(objects[0].value, 0);
  EXPECT_EQ(objects[1].value, 0);
}
