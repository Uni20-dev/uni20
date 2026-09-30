#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/async/linear_solve.hpp>
#include <uni20/linalg/async/matrix_product.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/tensor/async.hpp>

using namespace uni20;

TEST(MpfrLinalg, ProductAndSolveUseTensorDefaults)
{
  auto p = Precision::bits(256);
  DenseMatrix<mpreal> a(2, 2, p), x(2, 1, p), b;
  a[0, 0] = mpreal(0, p);
  a[1, 0] = mpreal(2, p);
  a[0, 1] = mpreal(1, p);
  a[1, 1] = mpreal(3, p);
  x[0, 0] = mpreal(1, p);
  x[1, 0] = mpreal(2, p);
  linalg::assign_product(b, a, x);
  EXPECT_EQ(b.default_precision(), p);
  EXPECT_EQ((b[0, 0]), 2);
  EXPECT_EQ((b[1, 0]), 8);
  auto solution = linalg::solve(a, b);
  EXPECT_EQ(solution.default_precision(), p);
  EXPECT_EQ((solution[0, 0]), 1);
  EXPECT_EQ((solution[1, 0]), 2);
  EXPECT_EQ((a[0, 0]), 0); // Preserving solve.
}

TEST(MpfrLinalg, ExplicitPrecisionOverridesDifferingDefaults)
{
  auto p = Precision::bits(256), q = Precision::bits(80);
  DenseMatrix<mpreal> a(1, 1, p), b(1, 1, q), out;
  a[0, 0] = mpreal(2, p);
  b[0, 0] = mpreal(6, q);
  EXPECT_THROW(linalg::assign_product(out, a, b), std::invalid_argument);
  EXPECT_THROW((void)linalg::solve(a, b), std::invalid_argument);
  linalg::assign_product(out, a, b, p);
  EXPECT_EQ((out[0, 0]), 12);
  auto result = linalg::solve(a, b, p);
  EXPECT_EQ((result[0, 0]), 3);
  EXPECT_EQ(result.default_precision(), p);
  auto info = linalg::solve_inplace_with_info(a, b, p);
  EXPECT_TRUE(info.succeeded());
  EXPECT_EQ(b.default_precision(), p);
}

TEST(MpfrLinalg, ConjugateAccessorIsObservedDuringPacking)
{
  auto p = Precision::bits(256);
  DenseMatrix<complex<mpreal>> a(1, 1, p), out;
  a[0, 0] = complex<mpreal>("3", "4", p);
  auto conjugated = conj(a);
  linalg::assign_product(out, a, conjugated);
  EXPECT_EQ((out[0, 0]), complex<mpreal>("25", "0", p));
  auto result = linalg::solve(conjugated, a);
  auto expected = a[0, 0] / conj(a[0, 0]);
  EXPECT_EQ((result[0, 0]), expected);
}

TEST(MpfrLinalg, GemmNoReadCasesAndOutputPrecision)
{
  auto p = Precision::bits(256), q = Precision::bits(80);
  DenseMatrix<mpreal> a(uninitialized, 2, 2, p), b(uninitialized, 2, 1, p), out(2, 1, q);
  out[0, 0] = mpreal(7, q);
  out[1, 0] = mpreal(9, q);
  EXPECT_THROW(linalg::gemm(out, mpreal(0, p), a, b, mpreal(1, p)), std::invalid_argument);
  linalg::gemm(out, mpreal(0, p), a, b, mpreal(2, p), p);
  EXPECT_EQ((out[0, 0]), 14);
  EXPECT_EQ((out[1, 0]), 18);
  EXPECT_EQ(out.default_precision(), p);
  out[0, 0] = mpreal{};
  out[1, 0] = mpreal{};
  linalg::gemm(out, mpreal(0, p), a, b, mpreal(0, p));
  EXPECT_EQ((out[0, 0]), 0);
  EXPECT_EQ((out[1, 0]), 0);
}

TEST(MpfrLinalg, RowMajorInputsProduceColumnMajorPreservingSolutions)
{
  auto p = Precision::bits(256);
  RowMajorTensor<mpreal, 2> a(2, 2, p), b(2, 1, p);
  a[0, 0] = mpreal(2, p);
  a[1, 1] = mpreal(4, p);
  b[0, 0] = mpreal(6, p);
  b[1, 0] = mpreal(8, p);
  auto x = linalg::solve(a, b, p);
  static_assert(std::same_as<typename decltype(x)::layout_type, ColumnMajor>);
  EXPECT_EQ((x[0, 0]), 3);
  EXPECT_EQ((x[1, 0]), 2);
  DenseMatrix<mpreal> result;
  linalg::assign_product(result, a, x);
  EXPECT_EQ((result[0, 0]), 6);
  EXPECT_EQ((result[1, 0]), 8);
}

TEST(MpfrLinalg, DiagnosticsAndEmptyNoops)
{
  auto p = Precision::bits(256);
  DenseMatrix<mpreal> a(2, 2, p), b(2, 1, p);
  a[0, 0] = mpreal(1, p);
  auto info = linalg::solve_inplace_with_info(a, b);
  EXPECT_EQ(info.status, linalg::SolveStatus::singular);
  EXPECT_EQ(info.pivot, 1);
  a[1, 1] = mpreal("1e-50", p);
  info = linalg::solve_inplace_with_info(a, b,
                                         linalg::SolveOptions<mpreal>{.relative_pivot_tolerance = mpreal("1e-40", p)});
  EXPECT_EQ(info.status, linalg::SolveStatus::small_pivot);
  a[0, 0] = mpreal("nan", p);
  info = linalg::solve_inplace_with_info(a, b);
  EXPECT_EQ(info.status, linalg::SolveStatus::nonfinite_input);
  DenseMatrix<mpreal> empty(2, 0, p);
  info = linalg::solve_inplace_with_info(a, empty);
  EXPECT_TRUE(info.succeeded());
}

TEST(MpfrLinalg, SolveResolvesAProblemThatRoundsToSingularAtBinary128Precision)
{
  auto p = Precision::bits(400), check = Precision::bits(512);
  auto epsilon = pow(mpreal(2, p), mpreal(-200, p));
  DenseMatrix<mpreal> a(2, 2, p), b(2, 1, p);
  a[0, 0] = mpreal(1, p);
  a[1, 0] = mpreal(1, p);
  a[0, 1] = mpreal(1, p);
  a[1, 1] = 1 + epsilon;
  b[0, 0] = mpreal(3, p);
  b[1, 0] = 3 + 2 * epsilon;
  auto x = linalg::solve(a, b);
  EXPECT_EQ((x[0, 0]), 1);
  EXPECT_EQ((x[1, 0]), 2);
  // Independent scalar residual at a higher precision, without provider GEMM.
  for (int i = 0; i != 2; ++i)
  {
    auto residual = a[i, 0].at(check) * x[0, 0].at(check) + a[i, 1].at(check) * x[1, 0].at(check) - b[i, 0].at(check);
    EXPECT_EQ(residual, 0);
  }
  auto info = linalg::solve_inplace_with_info(a, b, Precision::bits(113));
  EXPECT_EQ(info.status, linalg::SolveStatus::singular);
}

TEST(MpfrLinalg, AsyncProductsAndSolvesSelectPrecisionAfterAwaitingInputs)
{
  using namespace uni20::async;
  using Matrix = DenseMatrix<complex<mpreal>>;
  TbbScheduler scheduler{4};
  ScopedScheduler scope(&scheduler);
  std::vector<Async<Matrix>> solutions;
  for (int bits : {80, 256, 400, 512})
  {
    auto p = Precision::bits(bits);
    Async<Matrix> a, x, product;
    // Schedule consumers before publishing the two inputs.
    linalg::assign_product(product, a, x);
    solutions.push_back(linalg::solve(a, product));
    scheduler.schedule([](WriteBuffer<Matrix> a, WriteBuffer<Matrix> x, Precision p) static -> AsyncTask {
      Matrix av(1, 1, p), xv(1, 1, p);
      av[0, 0] = complex<mpreal>("1", "0", p);
      auto delta = pow(mpreal(2, p), mpreal(-p.bit_count() + 10, p));
      xv[0, 0] = complex<mpreal>(mpreal(1, p) + delta, mpreal(2, p));
      co_await a = std::move(av);
      co_await x = std::move(xv);
    }(a.write(), x.write(), p));
  }
  std::size_t i = 0;
  for (int bits : {80, 256, 400, 512})
  {
    auto p = Precision::bits(bits);
    auto const& result = solutions[i++].get_wait();
    EXPECT_EQ(result.default_precision(), p);
    EXPECT_EQ((result[0, 0].real() - 1), pow(mpreal(2, p), mpreal(-bits + 10, p)));
    EXPECT_EQ((result[0, 0].imag()), 2);
  }
}

TEST(MpfrLinalg, AsyncReservedParentAliasesAcceptExplicitOperationPrecision)
{
  using namespace uni20::async;
  using Matrix = DenseMatrix<complex<mpreal>>;
  DebugScheduler scheduler;
  ScopedScheduler scope(&scheduler);
  auto p = Precision::bits(256), q = Precision::bits(80);
  Async<Matrix> a, b, product;
  auto conjugated = uni20::async::conj(a);
  linalg::assign_product(product, conjugated, b, p);
  auto result = linalg::solve(conjugated, product, p);
  Matrix av(1, 1, q), bv(1, 1, q);
  av[0, 0] = complex<mpreal>("3", "4", q);
  bv[0, 0] = complex<mpreal>("1", "2", q);
  scheduler.schedule([](WriteBuffer<Matrix> a, WriteBuffer<Matrix> b, Matrix av, Matrix bv) static -> AsyncTask {
    co_await a = std::move(av);
    co_await b = std::move(bv);
  }(a.write(), b.write(), std::move(av), std::move(bv)));
  auto const& value = result.get_wait(scheduler);
  EXPECT_EQ(value.default_precision(), p);
  EXPECT_EQ((value[0, 0]), complex<mpreal>("1", "2", p));
}

TEST(MpfrLinalg, AsyncViewsInferPrecisionFromEachReadableParentEpoch)
{
  using namespace uni20::async;
  using Matrix = DenseMatrix<complex<mpreal>>;
  DebugScheduler scheduler;
  ScopedScheduler scope(&scheduler);
  auto p = Precision::bits(256), q = Precision::bits(80);
  Async<Matrix> a, b, first, second;
  auto producer_a = a.write();
  auto producer_b = b.write();
  auto conjugated = uni20::async::conj(a);
  auto reshaped = uni20::async::reshape_view(conjugated, 1, 1);
  auto rhs = uni20::async::reshape_view(std::as_const(b), 1, 1);
  linalg::assign_product(first, reshaped, rhs);
  auto first_solution = linalg::solve(reshaped, first);

  // This write belongs after the first consumers, but before the second ones.
  scheduler.schedule([](WriteBuffer<Matrix> a, WriteBuffer<Matrix> b, Precision q) static -> AsyncTask {
    auto av = co_await a;
    auto bv = co_await b;
    av.get().default_precision(q);
    bv.get().default_precision(q);
  }(a.write(), b.write(), q));
  linalg::assign_product(second, reshaped, rhs);
  auto second_solution = linalg::solve(reshaped, second);

  // Publish only after both sets of consumers have been scheduled.
  scheduler.schedule([](WriteBuffer<Matrix> a, WriteBuffer<Matrix> b, Precision p) static -> AsyncTask {
    Matrix av(1, 1, p), bv(1, 1, p);
    av[0, 0] = complex<mpreal>("3", "4", p);
    bv[0, 0] = complex<mpreal>("1", "2", p);
    co_await a = std::move(av);
    co_await b = std::move(bv);
  }(std::move(producer_a), std::move(producer_b), p));

  auto const& before = first_solution.get_wait(scheduler);
  auto const& after = second_solution.get_wait(scheduler);
  EXPECT_EQ(before.default_precision(), p);
  EXPECT_EQ(after.default_precision(), q);
  EXPECT_EQ((before[0, 0]), complex<mpreal>("1", "2", p));
  EXPECT_EQ((after[0, 0]), complex<mpreal>("1", "2", q));
  EXPECT_EQ(first.get_wait(scheduler).default_precision(), p);
  EXPECT_EQ(second.get_wait(scheduler).default_precision(), q);
}

TEST(MpfrLinalg, WritingThroughViewsPreservesParentDefault)
{
  auto p = Precision::bits(256), q = Precision::bits(80);
  DenseMatrix<mpreal> a(1, 1, p), b(1, 1, p), output(1, 1, q);
  a[0, 0] = mpreal(2, p);
  b[0, 0] = mpreal(3, p);
  auto view = reshape_view(output, 1, 1);
  linalg::assign_product(view, a, b);
  EXPECT_EQ((output[0, 0]), 6);
  EXPECT_EQ((output[0, 0].precision()), p);
  EXPECT_EQ(output.default_precision(), q);
  EXPECT_EQ(view.default_precision(), q);
  linalg::gemm(view, mpreal(1, p), a, b, mpreal(0, p));
  EXPECT_EQ(view.default_precision(), q);
  output.default_precision(p);
  linalg::add_product(view, a, b);
  EXPECT_EQ((output[0, 0]), 12);
}
