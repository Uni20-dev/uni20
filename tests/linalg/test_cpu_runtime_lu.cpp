#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/linalg/async/linear_solve.hpp>
#include <uni20/linalg/ops/slogdet.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/reshape.hpp>

using namespace uni20;
using namespace uni20::linalg;

namespace
{
struct UnexpectedFallback
{
    static constexpr std::string_view name = "unexpected_fallback";
    bool* called;
};
template <class Op, class... Args> consteval auto kernel_accepts_types(UnexpectedFallback const&, Op const&, Args&&...)
{
  return kernel_types_yes;
}
template <class Op, class... Args> KernelAttempt try_kernel(UnexpectedFallback backend, Op const&, Args&&...)
{
  *backend.called = true;
  return KernelAttempt::success;
}

template <class S> class CpuRuntimeLu : public testing::Test {};
using RuntimeScalars = testing::Types<mpreal, complex<mpreal>>;
TYPED_TEST_SUITE(CpuRuntimeLu, RuntimeScalars);

template <class S> S phase()
{
  if constexpr (Complex<S>)
    return S(mpreal{1}, mpreal{1});
  else
    return S{1};
}

// Exact integer data with two row swaps, two RHS columns and non-real complex
// pivots. Conversion starts with deliberately different element precisions.
template <class S> auto system(Precision p)
{
  DenseMatrix<S, RowMajor> a(3, 3, p), b(3, 2, p);
  int entries[3][3] = {{1, 1, 1}, {2, 2, 3}, {3, 4, 5}};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
    {
      auto value = S(entries[i][j]) * phase<S>();
      a[i, j] = j == 0 ? value : scalar_at_precision<S>(value, Precision::bits(j == 1 ? 64 : 320));
      b[i, 0] += value * S(j + 1);
      b[i, 1] += value * S(j == 0 ? -1 : j == 1 ? 0 : 2);
    }
  return std::pair{std::move(a), std::move(b)};
}

template <class Matrix> void check_solution(Matrix const& x, Precision p)
{
  auto tolerance = 128 * numeric_limits<mpreal>::epsilon(p);
  for (std::size_t i = 0; i < 3; ++i)
  {
    EXPECT_LE(math::abs(x[i, 0] - typename Matrix::value_type(i + 1)), tolerance);
    EXPECT_LE(math::abs(x[i, 1] - typename Matrix::value_type(i == 0 ? -1 : i == 1 ? 0 : 2)), tolerance);
    EXPECT_EQ(precision_of(x[i, 0]), p);
    EXPECT_EQ(precision_of(x[i, 1]), p);
  }
}

TYPED_TEST(CpuRuntimeLu, ReconstructReuseAndOneShotAtExplicitPrecision)
{
  using S = TypeParam;
  for (int bits : {128, 256})
  {
    auto p = Precision::bits(bits), q = Precision::bits(512), input_p = Precision::bits(80);
    auto [a, b] = system<S>(input_p);
    auto result = lu_factor_with_info(CpuReferenceBackend{}, a, p);
    ASSERT_TRUE(result.info.succeeded());
    ASSERT_TRUE(result.factor);
    auto const& f = *result.factor;
    EXPECT_EQ(f.precision(), p);
    ASSERT_TRUE(f.norms());
    EXPECT_EQ(f.norms()->one.precision(), p);
    auto pa = make_tensor<ColumnMajor>(a);
    for (std::size_t k = 0; k < 3; ++k)
      for (std::size_t j = 0; j < 3; ++j)
        std::swap(pa[k, j], pa[f.pivots()[k], j]);
    auto const& lu = f.packed();
    auto tolerance = scalar_at_precision<mpreal>(512 * numeric_limits<mpreal>::epsilon(p), q);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
      {
        auto sum = scalar_at_precision<S>(0, q);
        for (std::size_t k = 0; k < 3; ++k)
        {
          auto l = scalar_at_precision<S>(i == k ? S{1} : i > k ? S(lu[i, k]) : S{0}, q);
          auto u = scalar_at_precision<S>(k <= j ? S(lu[k, j]) : S{0}, q);
          sum += l * u;
        }
        EXPECT_LE(math::abs(sum - scalar_at_precision<S>(pa[i, j], q)), tolerance);
        EXPECT_EQ(precision_of(lu[i, j]), p);
      }
    for (int repeat = 0; repeat < 2; ++repeat)
      check_solution(lu_solve(CpuReferenceBackend{}, f, b), p);
    check_solution(solve(CpuReferenceBackend{}, a, b, p), p);
    // Default dispatch must work with MPLAPACK disabled too.
    check_solution(solve(a, b, p), p);
    auto inplace_a = a, inplace_b = b;
    ASSERT_TRUE(solve_inplace_with_info(CpuReferenceBackend{}, inplace_a, inplace_b, p).succeeded());
    check_solution(inplace_b, p);
    EXPECT_EQ(inplace_a.default_precision(), p);
    EXPECT_EQ(inplace_b.default_precision(), p);
    EXPECT_EQ(a.default_precision(), input_p);
    EXPECT_TRUE((a[0, 0].is_exact()));
    EXPECT_EQ(precision_of(a[0, 1]), Precision::bits(64));
    auto determinant = slogdet(f);
    S exact_det = -phase<S>() * phase<S>() * phase<S>();
    auto expected = scalar_at_precision<S>(exact_det, q);
    auto magnitude = math::abs(expected);
    EXPECT_LE(math::abs(scalar_at_precision<S>(determinant.phase, q) - expected / magnitude), tolerance);
    EXPECT_LE(math::abs(determinant.log_absolute.at(q) - math::log(magnitude)), tolerance);
    EXPECT_EQ(determinant.log_absolute.precision(), p);
#if UNI20_HAS_MPLAPACK_MPREAL
    // Portable factors can cross provider boundaries in either direction.
    check_solution(lu_solve(LapackBackend{}, f, b), p);
    auto provider_factors = lu_factor(LapackBackend{}, a, p);
    check_solution(lu_solve(CpuReferenceBackend{}, provider_factors, b), p);
#endif
  }
}

TYPED_TEST(CpuRuntimeLu, ExactInputsRequirePrecisionAndRoundBeforeFactorization)
{
  using S = TypeParam;
  auto p = Precision::bits(128);
  DenseMatrix<S> a(1, 1, Precision::exact()), b(1, 1, Precision::exact());
  a[0, 0] = S(mpreal(exact_rational("1/3")));
  b[0, 0] = S{1};
  EXPECT_THROW((void)lu_factor(CpuReferenceBackend{}, a), std::logic_error);
  EXPECT_THROW((void)solve(CpuReferenceBackend{}, a, b), std::logic_error);
  auto f = lu_factor(CpuReferenceBackend{}, a, p);
  EXPECT_EQ((f.packed()[0, 0]), scalar_at_precision<S>(a[0, 0], p));
  EXPECT_EQ(precision_of(f.packed()[0, 0]), p);
  auto x = lu_solve(CpuReferenceBackend{}, f, b);
  EXPECT_LE(math::abs(x[0, 0] - S{3}), 8 * numeric_limits<mpreal>::epsilon(p));
  EXPECT_EQ(precision_of(x[0, 0]), p);
  EXPECT_TRUE((a[0, 0].is_exact()));
  EXPECT_TRUE((b[0, 0].is_exact()));
}

TYPED_TEST(CpuRuntimeLu, FailureStatusesAndReusableFactors)
{
  using S = TypeParam;
  auto p = Precision::bits(128);
  DenseMatrix<S> a(2, 2, p), b(2, 1, p);
  a[0, 0] = S{1};
  bool fallback_called = false;
  auto backends = backend_list{CpuReferenceBackend{}, UnexpectedFallback{&fallback_called}};
  auto failure = lu_factor_with_info(backends, a);
  EXPECT_EQ(failure.info.status, SolveStatus::singular);
  EXPECT_EQ(failure.info.pivot, 1);
  EXPECT_FALSE(failure.factor);
  auto d = slogdet_with_info(CpuReferenceBackend{}, a);
  ASSERT_TRUE(d.value);
  EXPECT_EQ(d.value->phase, S{0});
  EXPECT_EQ(precision_of(d.value->phase), p);
  EXPECT_EQ(d.value->log_absolute, mpreal("-inf", p));
  EXPECT_EQ(solve_inplace_with_info(backends, a, b).status, SolveStatus::singular);
  EXPECT_FALSE(fallback_called);
  a[1, 1] = S(mpreal(exact_rational("1/100")));
  SolveOptions<mpreal> options{.relative_pivot_tolerance = mpreal(exact_rational("1/10"))};
  EXPECT_EQ(solve_inplace_with_info(CpuReferenceBackend{}, a, b, options).status, SolveStatus::small_pivot);
  EXPECT_EQ(lu_factor_with_info(CpuReferenceBackend{}, a,
                                LuOptions<mpreal>{.relative_pivot_tolerance = options.relative_pivot_tolerance})
                .info.status,
            SolveStatus::small_pivot);
  auto f = lu_factor(CpuReferenceBackend{}, a);
  b[0, 0] = scalar_at_precision<S>(mpreal("inf", p), p);
  EXPECT_EQ(lu_solve_inplace_with_info(CpuReferenceBackend{}, f, b).status, SolveStatus::nonfinite_input);
  EXPECT_FALSE(uni20::isfinite(b[0, 0]));
  b[0, 0] = S{2};
  auto x = lu_solve(CpuReferenceBackend{}, f, b);
  EXPECT_EQ((x[0, 0]), S{2});
  EXPECT_EQ((f.packed()[0, 0]), S{1});
  a[0, 0] = scalar_at_precision<S>(mpreal("nan", p), p);
  EXPECT_EQ(lu_factor_with_info(CpuReferenceBackend{}, a).info.status, SolveStatus::nonfinite_input);
  EXPECT_EQ(solve_inplace_with_info(CpuReferenceBackend{}, a, b).status, SolveStatus::nonfinite_input);
  DenseMatrix<S> no_rhs(2, 0, p), empty(0, 0, p);
  EXPECT_TRUE(solve_inplace_with_info(CpuReferenceBackend{}, a, no_rhs).succeeded());
  auto empty_f = lu_factor(CpuReferenceBackend{}, empty);
  EXPECT_EQ(empty_f.precision(), p);
  EXPECT_EQ(slogdet(empty_f).phase, scalar_at_precision<S>(1, p));
}

TYPED_TEST(CpuRuntimeLu, NonfiniteResultsAndConversionFailuresKeepFactorsReusable)
{
  using S = TypeParam;
  auto p = Precision::bits(128), high_p = Precision::bits(256);
  auto huge = math::ldexp(mpreal(1, p), mpfr_get_emax() - 1);
  ASSERT_TRUE(uni20::isfinite(huge));
  DenseMatrix<S> a(1, 1, p), b(1, 1, high_p);
  a[0, 0] = scalar_at_precision<S>(mpreal(exact_rational("1/4")), p);
  auto f = lu_factor(CpuReferenceBackend{}, a);
  b[0, 0] = scalar_at_precision<S>(huge, high_p);
  EXPECT_EQ(lu_solve_inplace_with_info(CpuReferenceBackend{}, f, b).status, SolveStatus::nonfinite_result);
  b[0, 0] = S{1};
  EXPECT_EQ((lu_solve(CpuReferenceBackend{}, f, b)[0, 0]), S{4});
  b[0, 0] = scalar_at_precision<S>(huge, high_p);
  EXPECT_EQ(solve_inplace_with_info(CpuReferenceBackend{}, a, b, p).status, SolveStatus::nonfinite_result);

  // A finite 256-bit input rounds out of range at 128 bits. No global
  // exponent/precision settings are changed to manufacture the conversion.
  auto before_rounding = math::ldexp(mpreal(2, high_p) - math::ldexp(mpreal(1, high_p), -200), mpfr_get_emax() - 1);
  ASSERT_TRUE(uni20::isfinite(before_rounding));
  ASSERT_FALSE(uni20::isfinite(before_rounding.at(p)));
  b[0, 0] = scalar_at_precision<S>(before_rounding, high_p);
  EXPECT_EQ(lu_solve_inplace_with_info(CpuReferenceBackend{}, f, b).status, SolveStatus::nonfinite_result);
  EXPECT_TRUE(uni20::isfinite(b[0, 0]));
  a[0, 0] = scalar_at_precision<S>(before_rounding, high_p);
  EXPECT_EQ(lu_factor_with_info(CpuReferenceBackend{}, a, p).info.status, SolveStatus::nonfinite_result);
  EXPECT_EQ(solve_inplace_with_info(CpuReferenceBackend{}, a, b, p).status, SolveStatus::nonfinite_result);
  EXPECT_TRUE(uni20::isfinite(a[0, 0]));

  DenseMatrix<S> overflow(2, 2, p);
  overflow[0, 0] = overflow[0, 1] = overflow[1, 1] = scalar_at_precision<S>(huge, p);
  overflow[1, 0] = scalar_at_precision<S>(-huge, p);
  EXPECT_EQ(lu_factor_with_info(CpuReferenceBackend{}, overflow).info.status, SolveStatus::nonfinite_result);
}

TYPED_TEST(CpuRuntimeLu, SolveViewsPreservesParentDefault)
{
  using S = TypeParam;
  auto p = Precision::bits(256), q = Precision::bits(80);
  DenseMatrix<S> a(1, 1, q), b(1, 1, q);
  a[0, 0] = S{2};
  b[0, 0] = S{6};
  auto av = reshape_view(a, 1, 1), bv = reshape_view(b, 1, 1);
  EXPECT_TRUE(solve_inplace_with_info(CpuReferenceBackend{}, av, bv, p).succeeded());
  EXPECT_EQ((b[0, 0]), S{3});
  EXPECT_EQ(precision_of(b[0, 0]), p);
  EXPECT_EQ(b.default_precision(), q);
  EXPECT_EQ(a.default_precision(), q);
  a[0, 0] = S{2} * phase<S>();
  auto f = lu_factor(CpuReferenceBackend{}, uni20::conj(a), p);
  b[0, 0] = S{8} * uni20::conj(phase<S>());
  EXPECT_TRUE(lu_solve_inplace_with_info(CpuReferenceBackend{}, f, bv).succeeded());
  EXPECT_EQ((b[0, 0]), S{4});
  EXPECT_EQ(b.default_precision(), q);
}

TYPED_TEST(CpuRuntimeLu, ConcurrentAsyncSolvesKeepWorkingPrecision)
{
  using namespace uni20::async;
  using S = TypeParam;
  using Matrix = DenseMatrix<S>;
  TbbScheduler scheduler{4};
  ScopedScheduler scope(&scheduler);
  std::vector<Async<Matrix>> solutions;
  for (int bits : {80, 128, 256, 400})
  {
    auto p = Precision::bits(bits);
    Async<Matrix> a, b;
    solutions.push_back(solve(CpuReferenceBackend{}, a, b));
    scheduler.schedule([](WriteBuffer<Matrix> a, WriteBuffer<Matrix> b, Precision p) static -> AsyncTask {
      Matrix av(1, 1, p), bv(1, 1, p);
      av[0, 0] = S{3};
      bv[0, 0] = S{1};
      co_await a = std::move(av);
      co_await b = std::move(bv);
    }(a.write(), b.write(), p));
  }
  std::size_t i = 0;
  for (int bits : {80, 128, 256, 400})
  {
    auto p = Precision::bits(bits);
    auto const& x = solutions[i++].get_wait();
    EXPECT_EQ(x.default_precision(), p);
    EXPECT_EQ(precision_of(x[0, 0]), p);
    EXPECT_EQ((x[0, 0]), scalar_at_precision<S>(mpreal(exact_rational("1/3")), p));
  }
}
} // namespace
