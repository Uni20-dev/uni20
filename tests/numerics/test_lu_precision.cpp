#include "precision_registry.hpp"
#include <uni20/linalg/ops/slogdet.hpp>

namespace uni20::test
{
// Force one provider: a successful CPU fallback must not certify its adapter.
template <class C> auto lu_provider()
{
#if UNI20_ENABLE_MPLAPACK_MPFR
  if constexpr (C::runtime)
    return linalg::MplapackMpfrBackend{};
  else
#endif
#if UNI20_ENABLE_MPLAPACK_BINARY80
      if constexpr (C::binary80_provider)
    return linalg::MplapackBinary80Backend{};
  else
#endif
    return linalg::LapackBackend{};
}

template <class C, class Backend> void check_lu_gap(Backend backend)
{
  testing::Test::RecordProperty("backend", std::string(Backend::name));
  auto a = C::matrix(2, 2), b = C::matrix(2, 2);
  auto phase = C::scalar(1, 1);
  a[0, 0] = a[0, 1] = a[1, 0] = phase;
  a[1, 1] = C::scalar(C::real(1) + C::gap(), C::real(0)) * phase;
  // Rounding the gap away makes A singular. Two RHS columns and a second
  // solve exercise reuse independently of the one-shot solve implementation.
  for (index_type col = 0; col < 2; ++col)
  {
    b[0, col] = C::scalar(col + 3) * phase;
    b[1, col] = C::scalar(C::real(col + 3) + C::real(col + 2) * C::gap(), C::real(0)) * phase;
  }
  auto attempt = linalg::lu_factor_with_info(backend, a);
  ASSERT_TRUE(attempt.info.succeeded());
  ASSERT_TRUE(attempt.factor);
  auto const& factors = *attempt.factor;
  for (int repeat = 0; repeat < 2; ++repeat)
  {
    auto x = linalg::lu_solve(backend, factors, b);
    for (index_type col = 0; col < 2; ++col)
    {
      expect_error_at_most(typename C::scalar_type(x[0, col]), C::scalar(1), C::real(16) * C::epsilon());
      expect_error_at_most(typename C::scalar_type(x[1, col]), C::scalar(col + 2), C::real(16) * C::epsilon());
      for (index_type row = 0; row < 2; ++row)
      {
        C::expect_precision(x[row, col]);
        C::expect_precision(factors.packed()[row, col]);
        expect_error_at_most(a[row, 0] * x[0, col] + a[row, 1] * x[1, col], typename C::scalar_type(b[row, col]),
                             C::real(64) * C::epsilon());
      }
    }
  }
}

UNI20_PRECISION_TEST(NumericalLinalg, CpuLuResolvesSmallGap) { check_lu_gap<TypeParam>(linalg::CpuReferenceBackend{}); }
UNI20_PRECISION_TEST(NumericalLinalg, ProviderLuResolvesSmallGap) { check_lu_gap<TypeParam>(lu_provider<TypeParam>()); }

#if UNI20_ENABLE_MPFR
template <class C, class Backend> void check_lu_accuracy(Backend backend)
{
  testing::Test::RecordProperty("backend", std::string(Backend::name));
  auto a = C::matrix(2, 2), b = C::matrix(2, 1);
  a[0, 0] = C::scalar(2);
  a[0, 1] = a[1, 0] = C::scalar(1);
  a[1, 1] = C::scalar(3);
  b[0, 0] = C::scalar(1, 1);
  auto factors = linalg::lu_factor(backend, a);
  auto x = linalg::lu_solve(backend, factors, b);
  for (index_type row = 0; row < 2; ++row)
  {
    int numerator = row == 0 ? 3 : -1;
    if constexpr (C::is_complex)
    {
      expect_rational_accuracy<C>(x[row, 0].real(), numerator, 5);
      expect_rational_accuracy<C>(x[row, 0].imag(), numerator, 5);
    }
    else
      expect_rational_accuracy<C>(x[row, 0], numerator, 5);
    C::expect_precision(x[row, 0]);
  }
}

UNI20_PRECISION_TEST(NumericalLinalg, CpuLuAccuracyImprovesWithPrecision)
{
  check_lu_accuracy<TypeParam>(linalg::CpuReferenceBackend{});
}
UNI20_PRECISION_TEST(NumericalLinalg, ProviderLuAccuracyImprovesWithPrecision)
{
  check_lu_accuracy<TypeParam>(lu_provider<TypeParam>());
}
#endif

template <class C, class Backend> void check_logdet_precision(Backend backend)
{
  testing::Test::RecordProperty("backend", std::string(Backend::name));
  auto a = C::matrix(1, 1);
  // For complex scalars use a purely imaginary diagonal, whose phase is
  // exactly i and whose log magnitude is still log(1 + gap).
  auto phase = C::is_complex ? C::scalar(0, 1) : C::scalar(-1);
  a[0, 0] = phase * C::scalar(C::real(1) + C::gap(), C::real(0));
  auto d = linalg::slogdet(backend, a);
  expect_equal(d.phase, phase);
  C::expect_precision(d.phase);
  C::expect_precision(d.log_absolute);
  // Independent Taylor bound: log(1+g) = g - g*g/2 + remainder,
  // with 0 <= remainder <= g*g*g/3. The quadratic term is dyadic and
  // representable here; dropping it (e.g. narrowing the small output) fails.
  auto quadratic = C::gap() * C::gap();
  auto reference = C::gap() - quadratic / C::real(2);
  auto tolerance = quadratic * C::gap() / C::real(3) + C::real(8) * C::epsilon() * C::gap();
  expect_error_at_most(d.log_absolute, reference, tolerance);

  // Retain a sub-leading component of a nontrivial complex unit phase too.
  if constexpr (C::is_complex)
  {
    a[0, 0] = C::scalar(C::real(1), C::real(1) + C::gap());
    d = linalg::slogdet(backend, a);
    auto ratio = d.phase.imag() / d.phase.real();
    expect_error_at_most(ratio, C::real(1) + C::gap(), C::real(8) * C::epsilon());
    C::expect_precision(d.phase);
  }
}

UNI20_PRECISION_TEST(NumericalLinalg, CpuLogDeterminantRetainsPrecision)
{
  check_logdet_precision<TypeParam>(linalg::CpuReferenceBackend{});
}
UNI20_PRECISION_TEST(NumericalLinalg, ProviderLogDeterminantRetainsPrecision)
{
  check_logdet_precision<TypeParam>(lu_provider<TypeParam>());
}

template <class C, class Backend> void check_logdet_range(Backend backend)
{
  testing::Test::RecordProperty("backend", std::string(Backend::name));
  auto a = C::matrix(2, 2);
  auto phase = C::is_complex ? C::scalar(0, 1) : C::scalar(1);
  for (int exponent : {-12000, -1200, 1200, 12000})
  {
    // A = scale*phase*[[1,1],[1,2]] requires elimination, not just a
    // diagonal scan. All exact factors remain representable even where an
    // unscaled complex division's squared denominator would overflow.
    a[0, 0] = a[0, 1] = a[1, 0] = C::scalar(C::power_of_two(exponent), C::real(0)) * phase;
    a[1, 1] = C::scalar(2) * a[0, 0];
    auto d = linalg::slogdet(backend, a);
    EXPECT_TRUE(uni20::isfinite(d.log_absolute));
    expect_equal(d.phase, typename C::scalar_type(phase * phase));
    C::expect_precision(d.phase);
    C::expect_precision(d.log_absolute);
    // Compare the logarithm after rescaling, without forming a determinant.
    expect_error_at_most(d.log_absolute / C::real(2 * exponent), math::log(C::real(2)), C::real(8) * C::epsilon());
  }
}

UNI20_PRECISION_TEST(NumericalLinalg, CpuLogDeterminantPreservesExtendedExponentRange)
{
  check_logdet_range<TypeParam>(linalg::CpuReferenceBackend{});
}
UNI20_PRECISION_TEST(NumericalLinalg, ProviderLogDeterminantPreservesExtendedExponentRange)
{
  check_logdet_range<TypeParam>(lu_provider<TypeParam>());
}
} // namespace uni20::test
