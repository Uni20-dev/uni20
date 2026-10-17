#include <uni20/config.hpp>

#if UNI20_HAS_MPREAL
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/async/linear_solve.hpp>
#include <uni20/linalg/async/matrix_product.hpp>
#include <uni20/linalg/backends/blas/gemm.hpp>
#include <uni20/linalg/backends/lapack/qr.hpp>
#include <uni20/linalg/ops/lu.hpp>
#include <utility>
#include <vector>

using namespace uni20;
using namespace uni20::linalg;

namespace
{
template <class S> consteval bool provider_acceptance_is_operation_specific()
{
  using Matrix = DenseMatrix<S>;
  using M = decltype(mdspec_of(std::declval<Matrix&>()));
  using R = decltype(mdspec_of(std::declval<Matrix const&>()));
  using Pivots = std::span<std::size_t>;
  using ConstPivots = std::span<std::size_t const>;
  using Options = SolveOptions<mpreal>;
  constexpr auto gemm = probe_dispatch_kernel_types<BlasBackend, gemm_op, M, S, R, R, S, Precision>();
  constexpr auto solve =
      probe_dispatch_kernel_types<LapackBackend, linear_solve_op, M, M, SolveInfo, Options, Precision>();
  constexpr auto factor =
      probe_dispatch_kernel_types<LapackBackend, lu_factor_op, M, Pivots, SolveInfo, Options, Precision>();
  constexpr auto reuse =
      probe_dispatch_kernel_types<LapackBackend, lu_solve_op, R, ConstPivots, M, SolveInfo, Precision>();
#if UNI20_HAS_MPLAPACK_MPREAL
  static_assert(gemm == KernelTypeAcceptance::maybe); // Exact arithmetic declines at runtime.
  static_assert(solve == KernelTypeAcceptance::yes);
  static_assert(factor == KernelTypeAcceptance::yes);
  static_assert(reuse == KernelTypeAcceptance::yes);
#else
  static_assert(gemm == KernelTypeAcceptance::no);
  static_assert(solve == KernelTypeAcceptance::no);
  static_assert(factor == KernelTypeAcceptance::no);
  static_assert(reuse == KernelTypeAcceptance::no);
#endif
  // Tensor-level support does not imply a native raw-pointer LAPACK ABI, and
  // raw runtime-scalar descriptors still require explicit working precision.
  static_assert(!LapackScalar<S>);
  static_assert(probe_dispatch_kernel_types<LapackBackend, qr_op, M, M, M>() == KernelTypeAcceptance::no);
#if UNI20_HAS_MPLAPACK_MPREAL
  static_assert(probe_dispatch_kernel_types<LapackBackend, qr_op, M, M, M, Precision>() == KernelTypeAcceptance::yes);
#else
  static_assert(probe_dispatch_kernel_types<LapackBackend, qr_op, M, M, M, Precision>() == KernelTypeAcceptance::no);
#endif
  static_assert(probe_dispatch_kernel_types<BlasBackend, gemm_op, M, S, R, R, S>() == KernelTypeAcceptance::no);
  static_assert(probe_dispatch_kernel_types<LapackBackend, linear_solve_op, M, M, SolveInfo, Options>() ==
                KernelTypeAcceptance::no);
  static_assert(probe_dispatch_kernel_types<LapackBackend, lu_factor_op, M, Pivots, SolveInfo, Options>() ==
                KernelTypeAcceptance::no);
  static_assert(probe_dispatch_kernel_types<LapackBackend, lu_solve_op, R, ConstPivots, M, SolveInfo>() ==
                KernelTypeAcceptance::no);
  return true;
}
static_assert(provider_acceptance_is_operation_specific<mpreal>());
#if UNI20_HAS_MPREAL
static_assert(provider_acceptance_is_operation_specific<complex<mpreal>>());
#endif

#if UNI20_HAS_MPLAPACK_MPREAL
template <class S> class MpfrBackendDispatch : public testing::Test {};
using Scalars = testing::Types<mpreal, complex<mpreal>>;
TYPED_TEST_SUITE(MpfrBackendDispatch, Scalars);

TYPED_TEST(MpfrBackendDispatch, DefaultSelectionUsesBlasAndLapackFamilies)
{
  using S = TypeParam;
  auto p = Precision::bits(192);
  RowMajorTensor<S, 2> a(2, 2, p);
  DenseMatrix<S> b(2, 1, p), product;
  a[0, 1] = S{1};
  a[1, 0] = S{2};
  a[1, 1] = S{3};
  auto gap = math::ldexp(mpreal(1, p), -160);
  b[0, 0] = scalar_at_precision<S>(1, p);
  b[1, 0] = scalar_at_precision<S>(mpreal(2, p) + gap, p);
  if constexpr (Complex<S>)
  {
    b[0, 0].imag(mpreal(1, p));
    b[1, 0].imag(mpreal(-1, p));
  }
  std::vector<std::pair<std::string, std::string>> selected;
  dispatch_diagnostics::scoped_sink capture([&](dispatch_diagnostics::event const& event) {
    if (event.operation == "gemm" || event.operation == "linear_solve" || event.operation == "lu_factor" ||
        event.operation == "lu_solve")
    {
      ASSERT_TRUE(event.selected_backend());
      selected.emplace_back(event.operation, *event.selected_backend());
    }
  });
  assign_product(product, a, b);
  EXPECT_EQ((product[0, 0]), (b[1, 0]));
  EXPECT_EQ((product[1, 0]), (2 * b[0, 0] + 3 * b[1, 0]));
  auto x = solve(a, product);
  auto factors = lu_factor(a);
  auto reused = lu_solve(factors, product);
  for (index_type i = 0; i < 2; ++i)
  {
    EXPECT_EQ((x[i, 0]), (b[i, 0]));
    EXPECT_EQ((reused[i, 0]), (b[i, 0]));
    EXPECT_EQ((x[i, 0].precision()), p);
  }
  std::vector<std::pair<std::string, std::string>> const expected{
      {"gemm", "blas"}, {"linear_solve", "lapack"}, {"lu_factor", "lapack"}, {"lu_solve", "lapack"}};
  EXPECT_EQ(selected, expected);
}

TYPED_TEST(MpfrBackendDispatch, ExplicitFamiliesWorkThroughAsync)
{
  using S = TypeParam;
  using Matrix = DenseMatrix<S>;
  async::DebugScheduler scheduler;
  async::ScopedScheduler scope(&scheduler);
  auto p = Precision::bits(192);
  Matrix av(1, 1, p), bv(1, 1, p);
  av[0, 0] = S{2};
  bv[0, 0] = S{3};
  async::Async<Matrix> a(std::move(av)), b(std::move(bv)), product;
  assign_product(BlasBackend{}, product, a, b, p);
  auto result = solve(LapackBackend{}, a, product, p);
  auto const& x = result.get_wait(scheduler);
  EXPECT_EQ((x[0, 0]), S{3});
  EXPECT_EQ(precision_of(x), p);
}
#endif
} // namespace
#endif
