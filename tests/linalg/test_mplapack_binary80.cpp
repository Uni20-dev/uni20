#include <gtest/gtest.h>
#include <uni20/linalg/async/linear_solve.hpp>
#include <uni20/linalg/async/matrix_product.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/linalg/ops/slogdet.hpp>
#include <uni20/tensor/conjugate.hpp>

using namespace uni20;
using namespace uni20::linalg;

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
  auto f = lu_factor(MplapackBinary80Backend{}, a);
  auto x = lu_solve(MplapackBinary80Backend{}, f, a);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_LT(abs(x[i, j] - identity[i, j]), 1e-18L);
  auto cpu = slogdet(CpuReferenceBackend{}, a);
  auto provider = slogdet(f);
  EXPECT_LT(abs(cpu.phase - provider.phase), 1e-18L);
  EXPECT_LT(abs(cpu.log_absolute - provider.log_absolute), 1e-18L);
  auto default_result = slogdet(a);
  EXPECT_LT(abs(default_result.phase - provider.phase), 1e-18L);
}

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
