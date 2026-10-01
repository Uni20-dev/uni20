#include <gtest/gtest.h>
#include <uni20/backend/mplapack/mpfr.hpp>
#include <uni20/backend/mplapack/precision_scope.hpp>
#include <uni20/core/math.hpp>

namespace
{
using Real = uni20::mpreal;
using Complex = uni20::complex<Real>;
using uni20::Precision;
namespace provider = uni20::mplapack;

TEST(MplapackMpfr, ScopeRestoresFirstUseAndNestedState)
{
  auto original_precision = mpfr_get_default_prec();
  auto original_rounding = mpfr_get_default_rounding_mode();
  auto original_options = mpfrxx::mpc_precision_override_storage();
  auto original_initialized = gmpfrxx_mkII::detail::mpfr_defaults_initialized_storage();
  {
    provider::detail::precision_scope outer(Precision::bits(80));
    EXPECT_EQ(mpfrxx::default_precision_bits(), 80);
    EXPECT_EQ(mpfrxx::default_mpc_real_precision_bits(), 80);
    EXPECT_EQ(mpfrxx::default_mpc_imag_precision_bits(), 80);
    try
    {
      provider::detail::precision_scope inner(Precision::bits(256));
      mpfrxx::mpfr_class x;
      mpfrxx::mpc_class z;
      EXPECT_EQ(mpfr_get_prec(x.mpfr_data()), 256);
      EXPECT_EQ(mpfr_get_prec(mpc_imagref(z.mpc_data())), 256);
      throw std::runtime_error("test exceptional scope exit");
    }
    catch (std::runtime_error const&)
    {}
    EXPECT_EQ(mpfrxx::default_precision_bits(), 80);
    EXPECT_EQ(mpfrxx::default_mpc_imag_precision_bits(), 80);
  }
  EXPECT_EQ(mpfr_get_default_prec(), original_precision);
  EXPECT_EQ(mpfr_get_default_rounding_mode(), original_rounding);
  EXPECT_EQ(mpfrxx::mpc_precision_override_storage().active, original_options.active);
  EXPECT_EQ(gmpfrxx_mkII::detail::mpfr_defaults_initialized_storage(), original_initialized);
}

TEST(MplapackMpfr, RealProductPreservesBitsBeyondBinary128AndDoesNotReadUnsetOutput)
{
  auto p = Precision::bits(400);
  auto delta = uni20::pow(Real(2, p), Real(-300, p));
  std::vector<Real> a{Real(1, p) + delta}, b{Real(1, p)}, c(1, Real(uni20::uninitialized));
  provider::gemm(1, 1, 1, Real(1, p), a, b, Real(0, p), c, p);
  EXPECT_EQ(c[0], a[0]);
  EXPECT_EQ(c[0] - 1, delta);
  EXPECT_EQ(c[0].precision(), p);
}

TEST(MplapackMpfr, PackingRoundsElementsAtTheSelectedPrecision)
{
  auto high = Precision::bits(256), low = Precision::bits(3);
  std::vector<Real> a{Real("1.125", high), Real("1.375", high)}, b{Real(1, low)}, c(2);
  provider::gemm(2, 1, 1, Real(1, high), a, b, Real(0, high), c, low);
  EXPECT_EQ(c[0], 1);
  EXPECT_EQ(c[1], Real("1.5", low));
  EXPECT_EQ(c[1].precision(), low);
  auto approximate = Real("0.1", low);
  a = {approximate};
  c.resize(1);
  provider::gemm(1, 1, 1, Real(1, high), a, b, Real(0, high), c, high);
  EXPECT_EQ(c[0], approximate.at(high));
  EXPECT_NE(c[0], Real("0.1", high));
}

TEST(MplapackMpfr, RealLuPivotsAndSolvesMultipleRightHandSides)
{
  auto p = Precision::bits(256);
  std::vector<Real> a{Real(0, p), Real(2, p), Real(1, p), Real(3, p)};
  std::vector<Real> b{Real(2, p), Real(8, p), Real(4, p), Real(18, p)};
  std::vector<std::size_t> pivots;
  EXPECT_EQ(provider::getrf(2, a, pivots, p), 0);
  ASSERT_EQ(pivots.size(), 2);
  EXPECT_EQ(pivots[0], 1);
  provider::getrs(2, 2, a, pivots, b, p);
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[1], 2);
  EXPECT_EQ(b[2], 3);
  EXPECT_EQ(b[3], 4);
}

TEST(MplapackMpfr, ComplexProductAndLuUseBothComponents)
{
  auto p = Precision::bits(256);
  std::vector<Complex> a{Complex("0", "0", p), Complex("2", "0", p), Complex("1", "1", p), Complex("3", "0", p)};
  std::vector<Complex> x{Complex("1", "2", p), Complex("3", "-1", p)}, b(2);
  auto one = Complex(Real(1, p)), zero = Complex(p);
  provider::gemm(2, 1, 2, one, a, x, zero, b, p);
  EXPECT_EQ(b[0], Complex("4", "2", p));
  EXPECT_EQ(b[1], Complex("11", "1", p));
  std::vector<std::size_t> pivots;
  EXPECT_EQ(provider::getrf(2, a, pivots, p), 0);
  provider::getrs(2, 1, a, pivots, b, p);
  EXPECT_EQ(b, x);
}

TEST(MplapackMpfr, SingularPivotAndInvalidMetadata)
{
  auto p = Precision::bits(256);
  std::vector<Real> a{Real(1, p), Real(2, p), Real(2, p), Real(4, p)};
  std::vector<std::size_t> pivots;
  EXPECT_EQ(provider::getrf(2, a, pivots, p), 2);
  std::vector<Real> b{Real(1, p), Real(2, p)};
  pivots[0] = 2;
  EXPECT_THROW(provider::getrs(2, 1, a, pivots, b, p), std::invalid_argument);
  EXPECT_THROW(provider::gemm(3, 1, 2, Real(1, p), a, b, Real(0, p), b, p), std::invalid_argument);
}

TEST(MplapackMpfr, CallsRestoreDistinctComplexDefaultsAndRoundingAfterFailure)
{
  provider::detail::precision_scope test_environment(Precision::bits(77));
  mpfrxx::set_default_rounding_mode(MPFR_RNDD);
  mpfrxx::set_default_mpc_precision_bits(91, 103);
  mpfrxx::set_default_mpc_rounding_mode(MPFR_RNDU, MPFR_RNDZ);
  auto p = Precision::bits(400);
  auto delta = uni20::pow(Real(2, p), Real(-300, p));
  std::vector<Complex> a{Complex(Real(1, p), Real(1, p) + delta)}, b{Complex(Real(1, p))}, c(1);
  provider::gemm(1, 1, 1, Complex(Real(1, p)), a, b, Complex(p), c, p);
  EXPECT_EQ(c[0], a[0]);
  EXPECT_EQ(c[0].imag() - 1, delta);
  a[0] = Complex(uni20::uninitialized);
  EXPECT_THROW(provider::gemm(1, 1, 1, Complex(Real(1, p)), a, b, Complex(p), c, p), std::logic_error);
  EXPECT_EQ(mpfrxx::default_precision_bits(), 77);
  EXPECT_EQ(mpfrxx::default_rounding_mode(), MPFR_RNDD);
  EXPECT_EQ(mpfrxx::default_mpc_real_precision_bits(), 91);
  EXPECT_EQ(mpfrxx::default_mpc_imag_precision_bits(), 103);
  EXPECT_EQ(mpfrxx::default_mpc_real_rounding_mode(), MPFR_RNDU);
  EXPECT_EQ(mpfrxx::default_mpc_imag_rounding_mode(), MPFR_RNDZ);
}
} // namespace
