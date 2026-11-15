#include "../numerics/precision_cases.hpp"
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async/contract.hpp>
#include <uni20/linalg/dispatch_diagnostics.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/reshape.hpp>

#include <array>
#include <string>
#include <vector>

using namespace uni20;
using namespace uni20::test;

namespace
{
constexpr std::array<std::pair<std::size_t, std::size_t>, 1> matrix_axes{{{1, 0}}};

TEST(ContractionPrecisionCoverage, ConfiguredBackends)
{
  testing::Test::RecordProperty("native",
                                "double explicit-precision validation; native contractions covered by existing suites");
  testing::Test::RecordProperty("mpfr_mpc_direct_looped_async_128_256", UNI20_HAS_MPLAPACK_MPREAL
                                                                            ? "supported: MPLAPACK GEMM"
                                                                            : "unavailable: MPLAPACK MPFR disabled");
  testing::Test::RecordProperty("runtime_cpu_exact",
                                UNI20_HAS_MPREAL ? "supported: CPU reference" : "unavailable: MPFR scalar disabled");
  testing::Test::RecordProperty("runtime_cpu_finite",
                                "unsupported: reference kernel does not convert operation precision");
}

TEST(ContractionPrecision, FixedResultRejectsIncompatiblePrecisionBeforeMutation)
{
  DenseMatrix<double> a(1, 1), b(1, 1), out(1, 1);
  a[0, 0] = 2;
  b[0, 0] = 3;
  out[0, 0] = 7;
  EXPECT_THROW(linalg::contract(out, 1., a, b, matrix_axes, 0., Precision::bits(128)), std::invalid_argument);
  EXPECT_EQ((out[0, 0]), 7);
  linalg::contract(out, 1., a, b, matrix_axes, 0., native_precision<double>());
  EXPECT_EQ((out[0, 0]), 6);
}

#if UNI20_HAS_MPREAL
TEST(ContractionPrecision, ExactReferenceAndFiniteDeclinePreserveTheirContracts)
{
  auto check = []<class S>() {
    auto const exact = Precision::exact();
    DenseMatrix<S> a(1, 1, exact), b(1, 1, exact), out(uninitialized, 1, 1, exact);
    a[0, 0] = S{exact_rational("1/3")};
    b[0, 0] = S{exact_rational("3/7")};
    auto ad = mdspec_of(std::as_const(a)), bd = mdspec_of(std::as_const(b));
    auto od = mdspec_of(out);
    auto op = linalg::contract_op<2, 2, 1>{.axes = linalg::make_contraction_axes<2, 2>(matrix_axes)};
    EXPECT_FALSE(
        linalg::try_dispatch_kernel(linalg::CpuReferenceBackend{}, op, od, S{1}, ad, bd, S{0}, Precision::bits(128)));
    EXPECT_FALSE((out[0, 0].initialized()));
    linalg::contract(linalg::CpuReferenceBackend{}, out, S{1}, a, b, matrix_axes, S{0});
    EXPECT_EQ((out[0, 0]), S{exact_rational("1/7")});
    EXPECT_TRUE((out[0, 0].is_exact()));
    EXPECT_EQ(out.default_precision(), exact);
  };
  check.operator()<mpreal>();
  check.operator()<complex<mpreal>>();
}
#endif

#if UNI20_HAS_MPLAPACK_MPREAL
using Cases = testing::Types<PrecisionCase<PrecisionKind::mp128, false>, PrecisionCase<PrecisionKind::mp128, true>,
                             PrecisionCase<PrecisionKind::mp256, false>, PrecisionCase<PrecisionKind::mp256, true>>;
template <class C> class MpfrContraction : public PrecisionTest<C> {};
TYPED_TEST_SUITE(MpfrContraction, Cases);

struct DispatchCounts
{
    int blas = 0, direct = 0, looped = 0;
    linalg::dispatch_diagnostics::scoped_sink sink{[this](auto const& event) {
      ASSERT_TRUE(event.succeeded());
      auto const backend = event.selected_backend();
      ASSERT_TRUE(backend);
      if (event.operation == "gemm")
      {
        EXPECT_EQ(*backend, "blas");
        ++blas;
      }
      else if (event.operation == "contract")
      {
        EXPECT_NE(*backend, "cpu_reference");
        direct += *backend == "direct_gemm_contraction";
        looped += *backend == "looped_gemm_contraction";
      }
    }};
};

TYPED_TEST(MpfrContraction, DirectGemmRetainsSubDoubleIncrementsAndIgnoresOldOutput)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  auto const p = C::precision();
  auto const gap = C::gap();
  S const input = C::scalar(C::real(1) + gap, gap);
  DenseMatrix<S> a(1, 1, p), b(1, 1, p), out(uninitialized, 1, 1, Precision::bits(32));
  a[0, 0] = input;
  b[0, 0] = C::scalar(1);
  DispatchCounts counts;
  linalg::contract(out, S{1}, conj(a), b, matrix_axes, S{0});
  EXPECT_EQ((out[0, 0]), uni20::conj(input));
  C::expect_precision(S(out[0, 0]));
  EXPECT_EQ(out.default_precision(), p);
  EXPECT_EQ(counts.direct, 1);
  EXPECT_EQ(counts.blas, 1);
}

TYPED_TEST(MpfrContraction, ExplicitPrecisionOverridesDefaultsAndRoundsInputs)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  auto const p = C::precision(), q = Precision::bits(32);
  DenseMatrix<S> a(1, 1, p), b(1, 1, q), out(uninitialized, 1, 1, q);
  a[0, 0] = C::scalar(C::real(1) + C::gap(), C::real(0));
  b[0, 0] = S{1};
  EXPECT_THROW(linalg::contract(out, S{1}, a, b, matrix_axes, S{0}), std::invalid_argument);
  EXPECT_FALSE((out[0, 0].initialized()));
  DispatchCounts counts;
  linalg::contract(out, S{1}, a, b, matrix_axes, S{0}, p);
  EXPECT_EQ((out[0, 0]), (a[0, 0]));
  EXPECT_EQ(out.default_precision(), p);
  b.default_precision(p);
  out.default_precision(q);
  EXPECT_THROW(linalg::contract(out, S{1}, a, b, matrix_axes, S{1}), std::invalid_argument);
  // A low explicit operation precision must round inputs, rather than computing
  // with their stored precision and merely relabeling the result.
  auto av = reshape_view(a, 1, 1), bv = reshape_view(b, 1, 1), ov = reshape_view(out, 1, 1);
  linalg::contract(ov, S{1}, av, bv, matrix_axes, S{0}, q);
  EXPECT_EQ((out[0, 0]), S{1});
  EXPECT_EQ((out[0, 0].precision()), q);
  EXPECT_EQ(a.default_precision(), p);
  EXPECT_NE((a[0, 0]), S{1});
  EXPECT_EQ(counts.blas, 2);
}

TYPED_TEST(MpfrContraction, LoopedGemmForwardsPrecisionAndBetaToEverySlice)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  auto const p = C::precision();
  using Strided = StridedTensor<S, 3>;
  Strided a(uninitialized, typename Strided::extents_type{2, 2, 3}, std::array<index_type, 3>{10, 3, 1});
  a.default_precision(p);
  RowMajorTensor<S, 2> b(3, 2, p);
  RowMajorTensor<S, 3> out(2, 2, 2, p);
  for (index_type i = 0; i < 2; ++i)
    for (index_type j = 0; j < 2; ++j)
      for (index_type k = 0; k < 3; ++k)
        a[i, j, k] = C::scalar(C::real(1 + 7 * i + 3 * j + k) + C::gap(), C::gap());
  for (index_type k = 0; k < 3; ++k)
    for (index_type n = 0; n < 2; ++n)
      b[k, n] = C::scalar(1);
  for (auto& value : out.storage())
    value = C::scalar(5);
  constexpr std::array<std::pair<std::size_t, std::size_t>, 1> axes{{{2, 0}}};
  DispatchCounts counts;
  auto normalized = linalg::make_contraction_axes<3, 2>(axes);
  linalg::contract(linalg::LoopedGemmContractionBackend{linalg::BlasBackend{}}, out, S{2}, a, b, normalized, S{3}, p);
  for (index_type i = 0; i < 2; ++i)
    for (index_type j = 0; j < 2; ++j)
      for (index_type n = 0; n < 2; ++n)
      {
        S const expected = C::scalar(C::real(27 + 42 * i + 18 * j) + 6 * C::gap(), 6 * C::gap());
        EXPECT_EQ((out[i, j, n]), expected);
        C::expect_precision(S(out[i, j, n]));
      }
  EXPECT_EQ(counts.looped, 1);
  EXPECT_EQ(counts.blas, 2);
}

TYPED_TEST(MpfrContraction, AsyncUsesPrecisionAfterAwaitingInputs)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  using Matrix = DenseMatrix<S>;
  auto const p = C::precision();
  async::DebugScheduler scheduler;
  async::ScopedScheduler scope(&scheduler);
  Matrix av(1, 1, p), bv(1, 1, p), ov(uninitialized, 1, 1, Precision::bits(32));
  av[0, 0] = C::scalar(C::real(1) + C::gap(), C::gap());
  S const expected = av[0, 0];
  bv[0, 0] = S{1};
  async::Async<Matrix> a, b, out(std::move(ov));
  auto producer_a = a.write();
  auto producer_b = b.write();
  DispatchCounts counts;
  linalg::contract(out, 1, a, b, matrix_axes, 0);
  // The input defaults do not exist when contract is scheduled.
  scheduler.schedule(
      [](async::WriteBuffer<Matrix> a, async::WriteBuffer<Matrix> b, Matrix av, Matrix bv) static -> async::AsyncTask {
        co_await a = std::move(av);
        co_await b = std::move(bv);
      }(std::move(producer_a), std::move(producer_b), std::move(av), std::move(bv)));
  auto const& result = out.get_wait(scheduler);
  EXPECT_EQ((result[0, 0]), expected);
  EXPECT_EQ(result.default_precision(), p);
  EXPECT_EQ(counts.blas, 1);
  linalg::contract(out, 1, a, b, matrix_axes, 0, Precision::bits(32));
  auto const& rounded = out.get_wait(scheduler);
  EXPECT_EQ((rounded[0, 0]), scalar_at_precision<S>(expected, Precision::bits(32)));
  EXPECT_EQ(rounded.default_precision(), Precision::bits(32));
  EXPECT_EQ(counts.blas, 2);
}
#endif
} // namespace
