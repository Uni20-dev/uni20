#include <uni20/config.hpp>

#if UNI20_ENABLE_MPFR
#include <gtest/gtest.h>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/async/matrix_product.hpp>
#include <uni20/tensor/async.hpp>

using namespace uni20;

template <class S> class ExactMatrixProduct : public testing::Test {};
using ExactScalars = testing::Types<mpreal
#if UNI20_ENABLE_MPC
                                    ,
                                    complex<mpreal>
#endif
                                    >;
TYPED_TEST_SUITE(ExactMatrixProduct, ExactScalars);

TYPED_TEST(ExactMatrixProduct, RationalProductAndAccumulationSelectCpu)
{
  using S = TypeParam;
  auto exact = Precision::exact();
  RowMajorTensor<S, 2> a(2, 2, exact);
  DenseMatrix<S> b(2, 2, exact), out(uninitialized, 2, 2, exact);
  a[0, 0] = S{exact_constant("1/3")};
  a[0, 1] = S{exact_constant("1/7")};
  a[1, 0] = S{exact_constant("-1/2")};
  a[1, 1] = S{exact_constant("2/5")};
  b[0, 0] = S{3};
  b[1, 1] = S{5};
  namespace diagnostics = linalg::dispatch_diagnostics;
  std::size_t calls = 0;
  diagnostics::scoped_sink capture([&](diagnostics::event const& event) {
    ++calls;
    ASSERT_TRUE(event.selected_backend());
    EXPECT_EQ(*event.selected_backend(), linalg::CpuReferenceBackend::name);
  });
  linalg::assign_product(out, a, b);
  EXPECT_EQ((out[0, 0]), S{1});
  EXPECT_EQ((out[0, 1]), S{exact_constant("5/7")});
  EXPECT_EQ((out[1, 0]), S{exact_constant("-3/2")});
  EXPECT_EQ((out[1, 1]), S{2});
  linalg::add_product(out, a, b);
  EXPECT_EQ((out[0, 1]), S{exact_constant("10/7")});
  EXPECT_EQ((out[1, 0]), S{-3});
  EXPECT_EQ(out.default_precision(), exact);
  for (auto const& value : out.storage())
    EXPECT_TRUE(value.is_exact());
  EXPECT_EQ(calls, 2);
}

TYPED_TEST(ExactMatrixProduct, NoReadCasesAndEmptyInnerDimension)
{
  using S = TypeParam;
  auto exact = Precision::exact();
  DenseMatrix<S> a(uninitialized, 1, 1, exact), b(uninitialized, 1, 1, exact);
  DenseMatrix<S> out(uninitialized, 1, 1, exact);
  linalg::gemm(out, 0, a, b, 1);
  EXPECT_FALSE((out[0, 0].initialized()));
  linalg::gemm(out, 0, a, b, 0);
  EXPECT_EQ((out[0, 0]), S{});
  EXPECT_TRUE((out[0, 0].is_exact()));
  DenseMatrix<S> empty_a(2, 0, exact), empty_b(0, 3, exact);
  linalg::assign_product(out, empty_a, empty_b);
  EXPECT_EQ(out.extent(0), 2);
  EXPECT_EQ(out.extent(1), 3);
  for (auto const& value : out.storage())
    EXPECT_EQ(value, S{});
}

TYPED_TEST(ExactMatrixProduct, RuntimeDeclinesDoNotTouchStorage)
{
  using S = TypeParam;
  auto exact = Precision::exact(), p = Precision::bits(80);
  DenseMatrix<S> a(uninitialized, 1, 1, exact), b(uninitialized, 1, 1, exact);
  DenseMatrix<S> out(uninitialized, 1, 1, exact);
  auto ad = mdspec_of(std::as_const(a)), bd = mdspec_of(std::as_const(b));
  auto od = mdspec_of(out);
  EXPECT_EQ(linalg::try_kernel(linalg::CpuReferenceBackend{}, linalg::gemm_op{}, od, S{1}, ad, bd, S{}, p),
            linalg::KernelAttempt::unsupported_instance);
#if UNI20_ENABLE_MPLAPACK_MPFR
  EXPECT_EQ(linalg::try_kernel(linalg::MplapackMpfrBackend{}, linalg::gemm_op{}, od, S{1}, ad, bd, S{}, exact),
            linalg::KernelAttempt::unsupported_instance);
#endif
  EXPECT_FALSE((out[0, 0].initialized()));
  EXPECT_EQ(out.default_precision(), exact);
}

TYPED_TEST(ExactMatrixProduct, AsyncProductRetainsExactArithmetic)
{
  using S = TypeParam;
  using Matrix = DenseMatrix<S>;
  auto exact = Precision::exact();
  async::DebugScheduler scheduler;
  async::ScopedScheduler scope(&scheduler);
  Matrix av(1, 1, exact), bv(1, 1, exact);
  av[0, 0] = S{exact_constant("1/3")};
  bv[0, 0] = S{exact_constant("3/7")};
  async::Async<Matrix> a(std::move(av)), b(std::move(bv)), out;
  linalg::assign_product(out, a, b);
  linalg::add_product(out, a, b);
  auto const& result = out.get_wait(scheduler);
  EXPECT_EQ((result[0, 0]), S{exact_constant("2/7")});
  EXPECT_TRUE((result[0, 0].is_exact()));
  EXPECT_EQ(result.default_precision(), exact);
}

#if UNI20_ENABLE_MPC
TEST(ExactComplexMatrixProduct, ConjugatingAccessorIsObserved)
{
  using C = complex<mpreal>;
  DenseMatrix<C> a(1, 1, Precision::exact()), out;
  a[0, 0] = C{mpreal{exact_constant("1/3")}, mpreal{exact_constant("2/7")}};
  linalg::assign_product(out, a, conj(a));
  EXPECT_EQ((out[0, 0]), C{exact_constant("85/441")});
  EXPECT_TRUE((out[0, 0].is_exact()));
}
#endif

#if UNI20_ENABLE_MPLAPACK_MPFR
TYPED_TEST(ExactMatrixProduct, FiniteOverrideUsesMplapack)
{
  using S = TypeParam;
  auto p = Precision::bits(3);
  DenseMatrix<S> a(1, 1, Precision::exact()), b(1, 1, Precision::exact()), out;
  a[0, 0] = S{exact_constant("9/8")};
  b[0, 0] = S{1};
  linalg::assign_product(out, a, b, p);
  EXPECT_EQ((out[0, 0]), S{1});
  EXPECT_EQ((out[0, 0].precision()), p);
  EXPECT_EQ(out.default_precision(), p);
}
#endif
#endif
