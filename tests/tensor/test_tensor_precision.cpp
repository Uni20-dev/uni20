#include <gtest/gtest.h>
#include <uni20/core/math.hpp>
#include <uni20/tensor/tensor.hpp>

using namespace uni20;
using namespace uni20::literals;

TEST(TensorPrecision, ConstructionAndElementPrecisionAreIndependent)
{
  auto p = Precision::bits(256), q = Precision::bits(80);
  DenseMatrix<mpreal> a(2, 3, p);
  EXPECT_EQ(a.default_precision(), p);
  for (auto const& x : a.storage())
  {
    EXPECT_EQ(x, 0);
    EXPECT_EQ(x.precision(), p);
  }
  a[0, 1] = mpreal("0.1", q);
  EXPECT_EQ((a[0, 1].precision()), q);
  EXPECT_EQ(a.default_precision(), p);
  a.default_precision(q);
  EXPECT_EQ((a[0, 0].precision()), p);
  EXPECT_EQ(a.default_precision(), q);
  a.reset_shape(decltype(a)::extents_type(3, 4));
  EXPECT_EQ(a.default_precision(), q);
  EXPECT_EQ((a[2, 3]), 0);
  EXPECT_EQ((a[2, 3].precision()), q);
}

TEST(TensorPrecision, UnsetStorageAndEmptyShapesCarryDefaults)
{
  auto p = Precision::bits(256);
  DenseMatrix<mpreal> a(uninitialized, 2, 3, p);
  EXPECT_FALSE((a[1, 2].initialized()));
  EXPECT_EQ(a.default_precision(), p);
  a[1, 2] = mpreal(7, p);
  auto copy = a;
  EXPECT_EQ((copy[1, 2]), 7);
  EXPECT_FALSE((copy[0, 0].initialized()));
  EXPECT_EQ(copy.default_precision(), p);
  DenseMatrix<mpreal> empty(0, 3, p);
  EXPECT_EQ(empty.default_precision(), p);
  auto moved = std::move(empty);
  EXPECT_EQ(moved.default_precision(), p);
  ScalarTensor<mpreal> scalar(p);
  EXPECT_EQ(scalar[], 0);
  EXPECT_EQ(scalar.default_precision(), p);
  HostBuffer<mpreal> buffer(1, StorageInitialization::Zero, p);
  buffer[0] = mpreal(3, p);
  buffer.resize(4, StorageInitialization::Zero);
  EXPECT_EQ(buffer[0], 3);
  EXPECT_EQ(buffer[3], 0);
  EXPECT_EQ(buffer[3].precision(), p);
}

TEST(TensorPrecision, UninitializedResetPreservesOptionalDefault)
{
  auto check = []<class S>() {
    DenseMatrix<S> a;
    using Extents = typename decltype(a)::extents_type;
    a.reset_shape(uninitialized, Extents(2, 3));
    EXPECT_FALSE(a.default_precision_if_set());
    for (auto const& value : a.storage()) EXPECT_FALSE(value.initialized());
    // Ordinary zero initialization still needs a default, and failure preserves a.
    EXPECT_THROW(a.reset_shape(Extents(3, 4)), std::logic_error);
    EXPECT_EQ(a.extent(0), 2);
    EXPECT_EQ(a.extent(1), 3);
    a.reset_shape(uninitialized, Extents(0, 3));
    EXPECT_FALSE(a.default_precision_if_set());
    for (auto p : {Precision::exact(), Precision::bits(80)})
    {
      a.default_precision(p);
      a.reset_shape(uninitialized, Extents(1, 2));
      EXPECT_EQ(a.default_precision(), p);
      for (auto const& value : a.storage()) EXPECT_FALSE(value.initialized());
    }
  };
  check.operator()<mpreal>();
#if UNI20_ENABLE_MPC
  check.operator()<complex<mpreal>>();
#endif
}

#if UNI20_ENABLE_MPC
TEST(TensorPrecision, ViewsFollowParentDefaultsAndMaterializationCopiesThem)
{
  auto p = Precision::bits(256), q = Precision::bits(80);
  DenseMatrix<complex<mpreal>> a(2, 3, p);
  a[0, 0] = complex<mpreal>("1", "2", p);
  auto c = conj(a);
  auto r = reshape_view(a, 6);
  a.default_precision(q);
  EXPECT_EQ(c.default_precision(), q);
  EXPECT_EQ(r.default_precision(), q);
  auto twice = conj(c);
  EXPECT_EQ(twice.default_precision(), q);
  auto copy = Tensor(c);
  EXPECT_EQ(copy.default_precision(), q);
  EXPECT_EQ((copy[0, 0]), complex<mpreal>("1", "-2", p));
  auto nested = conj(r);
  auto restored = conj(nested);
  a.default_precision(p);
  EXPECT_EQ(nested.default_precision(), p);
  EXPECT_EQ(restored.default_precision(), p);
  EXPECT_EQ(copy.default_precision(), q); // Materialization owns its metadata.
  a.default_precision(q);
  auto flat = reshape(std::move(a), 6);
  EXPECT_EQ(flat.default_precision(), q);
  EXPECT_EQ(flat[0].precision(), p);
}
#endif

TEST(TensorPrecision, RealViewsFollowParentDefaults)
{
  auto p = Precision::bits(256), q = Precision::bits(80);
  DenseMatrix<mpreal> a(2, 3, p);
  auto const& c = conj(a);
  IndirectReshapedTensorView<DenseMatrix<mpreal>, stdex::layout_left, int> r(a, 6);
  a.default_precision(q);
  EXPECT_EQ(c.default_precision(), q);
  EXPECT_EQ(r.default_precision(), q);
  auto copy = Tensor(r);
  EXPECT_EQ(copy.default_precision(), q);
  auto deferred = ConstTensorView<DenseMatrix<mpreal>>(&a);
  EXPECT_EQ(deferred.default_precision_if_set(), q);
  a.default_precision(p);
  EXPECT_EQ(deferred.default_precision(), p);
  EXPECT_EQ(r.default_precision(), p);
  EXPECT_EQ(copy.default_precision(), q);

  DenseMatrix<mpreal> empty;
  auto const& empty_view = conj(empty);
  EXPECT_FALSE(empty_view.default_precision_if_set());
  EXPECT_THROW((void)empty_view.default_precision(), std::logic_error);
  empty.default_precision(q);
  EXPECT_EQ(empty_view.default_precision(), q);
}

TEST(TensorPrecision, BulkConversionChangesValuesAndDefaultExplicitly)
{
  auto p = Precision::bits(256), low = Precision::bits(3);
  DenseMatrix<mpreal> a(2, 2, p);
  a[0, 0] = mpreal("1.125", p);
  a[1, 0] = mpreal("1.375", p);
  a[0, 1] = mpreal(8, low); // Mixed element precisions do not affect the default.
  auto b = at_precision(a, low);
  EXPECT_EQ((a[0, 0]), 1.125_mp);
  EXPECT_EQ(a.default_precision(), p);
  EXPECT_EQ((b[0, 0]), 1);
  EXPECT_EQ((b[1, 0]), 1.5_mp);
  EXPECT_EQ(b.default_precision(), low);
  EXPECT_THROW((void)common_default_precision(a, b), std::invalid_argument);
  b.default_precision(p); // Metadata change does not convert the elements.
  EXPECT_EQ(common_default_precision(a, b), p);
  EXPECT_EQ((b[0, 0].precision()), low);
}

TEST(TensorPrecision, OutputPreparationRecordsPrecisionWithoutConvertingReusedValues)
{
  auto p = Precision::bits(256), q = Precision::bits(80);
  DenseMatrix<mpreal> a(2, 2, p);
  auto* original = a.storage().data();
  prepare_output(a, a.extents(), q);
  EXPECT_EQ(a.storage().data(), original);
  EXPECT_EQ(a.default_precision(), q);
  EXPECT_EQ((a[0, 0].precision()), p);
  prepare_output(a, decltype(a)::extents_type(3, 2), p);
  EXPECT_EQ(a.default_precision(), p);
  EXPECT_FALSE((a[0, 0].initialized()));
  auto storage = async::make_shared_storage<DenseMatrix<mpreal>>();
  auto& result = prepare_output(storage, a.extents(), q);
  EXPECT_EQ(result.default_precision(), q);
  EXPECT_FALSE((result[0, 0].initialized()));
}
