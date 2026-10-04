#include <gtest/gtest.h>
#include <uni20/core/math.hpp>
#include <uni20/linalg/async/linear_solve.hpp>
#include <uni20/linalg/async/matrix_product.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/linalg/ops/slogdet.hpp>
#include <uni20/tensor/conjugate.hpp>

using namespace uni20;
using namespace uni20::linalg;

namespace
{
using ComplexDescriptor = decltype(mdspec_of(std::declval<DenseMatrix<complex160>&>()));
using ConstComplexDescriptor = decltype(mdspec_of(std::declval<DenseMatrix<complex160> const&>()));
constexpr auto complex_lu_acceptance =
    UNI20_HAS_MPLAPACK_BINARY80_COMPLEX_LU ? KernelTypeAcceptance::yes : KernelTypeAcceptance::no;
static_assert(probe_dispatch_kernel_types<MplapackBinary80Backend, lu_factor_op, ComplexDescriptor&,
                                          std::span<std::size_t>, SolveInfo&, SolveOptions<float80> const&>() ==
              complex_lu_acceptance);
static_assert(probe_dispatch_kernel_types<MplapackBinary80Backend, lu_solve_op, ConstComplexDescriptor&,
                                          std::span<std::size_t const>, ComplexDescriptor&, SolveInfo&>() ==
              complex_lu_acceptance);
static_assert(probe_dispatch_kernel_types<MplapackBinary80Backend, linear_solve_op, ComplexDescriptor&,
                                          ComplexDescriptor&, SolveInfo&, SolveOptions<float80> const&>() ==
              complex_lu_acceptance);

template <class Factor, class Solve, class SolveFromFactors>
void check_complex_exponent_range(Factor factorize, Solve solve_once, SolveFromFactors solve_factors)
{
  auto const epsilon = numeric_limits<float80>::epsilon();
  for (float80 value : {1e4000L, 1e-4000L})
  {
    DenseMatrix<complex160> a(1, 1), b(1, 1);
    a[0, 0] = {value, value};
    b[0, 0] = complex160{1};
    auto result = factorize(a);
    ASSERT_TRUE(result.info.succeeded());
    ASSERT_TRUE(result.factor.has_value());
    auto from_factors = solve_factors(*result.factor, b);
    auto once = solve_once(a, b);
    // The reciprocal is representable at both scales. Compare after scaling
    // back, so neither an absolute tolerance nor a double conversion hides zero.
    for (auto const& x : {from_factors[0, 0], once[0, 0]})
    {
      EXPECT_LE(math::abs(x.real() * value - 0.5L), 16 * epsilon);
      EXPECT_LE(math::abs(x.imag() * value + 0.5L), 16 * epsilon);
    }
  }
  for (float80 value : {1e3000L, 1e-3000L})
  {
    DenseMatrix<complex160> a(2, 2);
    a[0, 0] = a[0, 1] = a[1, 0] = complex160{value, value};
    a[1, 1] = {2 * value, 2 * value};
    auto result = factorize(a);
    ASSERT_TRUE(result.info.succeeded());
    ASSERT_TRUE(result.factor.has_value());
    auto const& f = *result.factor;
    EXPECT_LE(math::abs(f.packed()[1, 0] - complex160{1}), 16 * epsilon);
    auto determinant_result = slogdet_with_info(f);
    ASSERT_TRUE(determinant_result.info.succeeded());
    ASSERT_TRUE(determinant_result.value.has_value());
    auto const& determinant = *determinant_result.value;
    // A = value*(1+i)*[[1,1],[1,2]], hence det(A)=2*i*value^2.
    auto expected_log = 2 * math::log(value) + math::log(float80{2});
    EXPECT_LE(math::abs(determinant.phase - complex160{0, 1}), 32 * epsilon);
    EXPECT_LE(math::abs(determinant.log_absolute - expected_log), 64 * epsilon * math::abs(expected_log));
  }
}
} // namespace

TEST(Binary80Linalg, PreservesPrecisionAndExponentRange)
{
  // All of these values lose information if the adapter passes through double.
  for (float80 value : {1.0L + 0x1p-60L, 1e4000L, 1e-4000L})
  {
    DenseMatrix<float80> a(1, 1), b(1, 1), c(1, 1);
    a[0, 0] = value;
    b[0, 0] = 1;
    gemm(MplapackBinary80Backend{}, c, 1, a, b, 0);
    EXPECT_EQ((c[0, 0]), value);
    auto f = lu_factor(MplapackBinary80Backend{}, a);
    EXPECT_EQ((f.packed()[0, 0]), value);
    auto x = lu_solve(MplapackBinary80Backend{}, f, a);
    EXPECT_EQ((x[0, 0]), 1);
    auto once = solve(MplapackBinary80Backend{}, a, a);
    EXPECT_EQ((once[0, 0]), 1);
  }
  DenseMatrix<complex160> a(2, 2), identity(2, 2), product;
  a[0, 0] = {1.0L + 0x1p-60L, 2};
  a[0, 1] = complex160{3};
  a[1, 0] = {4, -1};
  a[1, 1] = {5, 2};
  identity[0, 0] = identity[1, 1] = complex160{1};
  assign_product(MplapackBinary80Backend{}, product, uni20::conj(a), identity);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_EQ((product[i, j]), uni20::conj(a[i, j]));
  auto cpu = slogdet(CpuReferenceBackend{}, a);
  auto default_result = slogdet(a);
  EXPECT_LT(abs(default_result.phase - cpu.phase), 1e-18L);
  EXPECT_LT(abs(default_result.log_absolute - cpu.log_absolute), 1e-18L);
#if UNI20_HAS_MPLAPACK_BINARY80_COMPLEX_LU
  auto f = lu_factor(MplapackBinary80Backend{}, a);
  auto x = lu_solve(MplapackBinary80Backend{}, f, a);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_LT(abs(x[i, j] - identity[i, j]), 1e-18L);
  auto provider = slogdet(f);
  EXPECT_LT(abs(cpu.phase - provider.phase), 1e-18L);
  EXPECT_LT(abs(cpu.log_absolute - provider.log_absolute), 1e-18L);
  EXPECT_LT(abs(default_result.phase - provider.phase), 1e-18L);
#endif
}

TEST(Binary80Linalg, ComplexDefaultSolveAndLogdetPreserveExponentRange)
{
  check_complex_exponent_range([](auto const& a) { return lu_factor_with_info(a); },
                               [](auto const& a, auto const& b) { return solve(a, b); },
                               [](auto const& f, auto const& b) { return lu_solve(f, b); });
}

#if UNI20_HAS_MPLAPACK_BINARY80_COMPLEX_LU
TEST(Binary80Linalg, ComplexProviderSolveAndLogdetPreserveExponentRange)
{
  check_complex_exponent_range([](auto const& a) { return lu_factor_with_info(MplapackBinary80Backend{}, a); },
                               [](auto const& a, auto const& b) { return solve(MplapackBinary80Backend{}, a, b); },
                               [](auto const& f, auto const& b) { return lu_solve(MplapackBinary80Backend{}, f, b); });
}
#endif

TEST(Binary80Linalg, ZeroCoefficientsAndEmptyInnerExtent)
{
  DenseMatrix<complex160> a(2, 2), b(2, 2), c(2, 2);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      a[i, j] = b[i, j] = c[i, j] = complex160{numeric_limits<float80>::quiet_NaN(), 0};
  gemm(MplapackBinary80Backend{}, c, 0, a, b, 0);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_EQ((c[i, j]), complex160{});
  a = DenseMatrix<complex160>(2, 0);
  b = DenseMatrix<complex160>(0, 2);
  c[0, 0] = {3, 4};
  gemm(MplapackBinary80Backend{}, c, 1, a, b, 2);
  EXPECT_EQ((c[0, 0]), (complex160{6, 8}));
}

TEST(Binary80Linalg, AsyncDeferredProductAndSolve)
{
  using namespace uni20::async;
  DebugScheduler scheduler;
  ScopedScheduler scope(&scheduler);
  DenseMatrix<float80> a(1, 1), b(1, 1);
  a[0, 0] = 2;
  b[0, 0] = 1 + 0x1p-60L;
  Async<DenseMatrix<float80>> av(std::move(a)), bv(std::move(b)), product;
  assign_product(MplapackBinary80Backend{}, product, av, bv);
  auto result = solve(MplapackBinary80Backend{}, av, product);
  EXPECT_EQ((result.get_wait()[0, 0]), 1 + 0x1p-60L);
}
