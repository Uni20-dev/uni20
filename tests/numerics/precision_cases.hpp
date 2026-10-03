#pragma once

#include <gtest/gtest.h>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>
#include <uni20/tensor/tensor.hpp>

#include <string>
#include <type_traits>

namespace uni20::test
{
template <class R, bool C> struct TestScalar
{
    using type = R;
};
template <class R> struct TestScalar<R, true>
{
    using type = uni20::complex<R>;
};
// A case is a test context, not a new production scalar type. In particular,
// both MPFR cases instantiate exactly the same runtime-precision algorithms.
template <class R, bool IsComplex, int Bits = 0> struct PrecisionContext
{
    using real_type = R;
    using scalar_type = typename TestScalar<R, IsComplex>::type;
    static constexpr bool available = true;
    static constexpr bool is_complex = IsComplex;
    static constexpr bool runtime = Bits != 0;

    static R real(int n)
    {
#if UNI20_ENABLE_MPFR
      if constexpr (runtime)
        return R(n, Precision::bits(Bits));
      else
#endif
        return R(n);
    }

    static constexpr int digits()
    {
      if constexpr (runtime)
        return Bits;
      else
        return numeric_limits<R>::digits;
    }

    static R epsilon() { return numeric_limits<R>::epsilon(real(1)); }

    static R power_of_two(int exponent)
    {
      R value = real(1);
      // Multiplication/division by two is exact in these modest test ranges.
      for (int i = 0; i < exponent; ++i)
        value *= real(2);
      for (int i = 0; i > exponent; --i)
        value /= real(2);
      return value;
    }

    // Leave enough guard bits for small dense algorithms, but exercise bits
    // beyond the immediately preceding scalar in the certification ladder.
    static R gap() { return power_of_two(8 - digits()); }

    static scalar_type scalar(R re, R im)
    {
      if constexpr (IsComplex)
        return scalar_type(re, im);
      else
        return re;
    }
    static scalar_type scalar(int re, int im = 0) { return scalar(real(re), real(im)); }

    static auto matrix(index_type rows, index_type cols)
    {
#if UNI20_ENABLE_MPFR
      if constexpr (runtime)
        return DenseMatrix<scalar_type>(rows, cols, Precision::bits(Bits));
      else
#endif
        return DenseMatrix<scalar_type>(rows, cols);
    }

    template <class T> static void expect_precision(T const& value)
    {
#if UNI20_ENABLE_MPFR
      if constexpr (runtime)
      {
        EXPECT_FALSE(value.is_exact());
        EXPECT_EQ(value.precision(), Precision::bits(Bits));
      }
#endif
    }
};

enum class PrecisionKind
{
  float32,
  float64,
  float80,
  float128,
  mp128,
  mp256
};

template <PrecisionKind Kind, bool IsComplex> struct CaseInfo
{
    // Explicit expected coverage, not inferred from the production concepts:
    // accidentally removing a supported scalar must fail compilation/testing.
    static constexpr bool projected_lapack =
        Kind == PrecisionKind::float32 || Kind == PrecisionKind::float64 || Kind == PrecisionKind::float128;
    // GEMM/solve provider coverage can grow independently of projected solvers.
    static constexpr bool binary80_provider = Kind == PrecisionKind::float80;

    static std::string name()
    {
      char const* names[] = {"Float32", "Float64", "Float80", "Float128", "Mp128", "Mp256"};
      return std::string(IsComplex ? "Complex" : "Real") + names[static_cast<int>(Kind)];
    }
};

// Keep unavailable cases in the coverage registry. They appear in the report,
// while only configured, applicable probes become executable tests.
template <PrecisionKind Kind, bool IsComplex> struct PrecisionCase : CaseInfo<Kind, IsComplex>
{
    static constexpr bool available = false;
};

#define UNI20_TEST_PRECISION_CASE(kind, real, bits)                                                                    \
  template <bool C>                                                                                                    \
  struct PrecisionCase<PrecisionKind::kind, C> : CaseInfo<PrecisionKind::kind, C>, PrecisionContext<real, C, bits>     \
  {}

UNI20_TEST_PRECISION_CASE(float32, float, 0);
UNI20_TEST_PRECISION_CASE(float64, double, 0);
#if UNI20_HAS_FLOAT80
UNI20_TEST_PRECISION_CASE(float80, uni20::float80, 0);
#endif
#if UNI20_HAS_FLOAT128
#if !defined(MPLAPACK_BINARY128_MODE) || MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL
UNI20_TEST_PRECISION_CASE(float128, uni20::float128, 0);
#endif
#endif
#if UNI20_ENABLE_MPFR
template <PrecisionKind K>
  requires(K == PrecisionKind::mp128 || K == PrecisionKind::mp256)
struct PrecisionCase<K, false> : CaseInfo<K, false>,
                                 PrecisionContext<mpreal, false, K == PrecisionKind::mp128 ? 128 : 256>
{};
#if UNI20_ENABLE_MPC
template <PrecisionKind K>
  requires(K == PrecisionKind::mp128 || K == PrecisionKind::mp256)
struct PrecisionCase<K, true> : CaseInfo<K, true>, PrecisionContext<mpreal, true, K == PrecisionKind::mp128 ? 128 : 256>
{};
#endif
#endif
#undef UNI20_TEST_PRECISION_CASE

using PrecisionCases =
    testing::Types<PrecisionCase<PrecisionKind::float32, false>, PrecisionCase<PrecisionKind::float32, true>,
                   PrecisionCase<PrecisionKind::float64, false>, PrecisionCase<PrecisionKind::float64, true>,
                   PrecisionCase<PrecisionKind::float80, false>, PrecisionCase<PrecisionKind::float80, true>,
                   PrecisionCase<PrecisionKind::float128, false>, PrecisionCase<PrecisionKind::float128, true>,
                   PrecisionCase<PrecisionKind::mp128, false>, PrecisionCase<PrecisionKind::mp128, true>,
                   PrecisionCase<PrecisionKind::mp256, false>, PrecisionCase<PrecisionKind::mp256, true>>;

template <class C> class PrecisionTest : public testing::Test {
  protected:
    void SetUp() override
    {
      this->RecordProperty("scalar", C::name());
      if constexpr (C::available) this->RecordProperty("bits", C::digits());
    }
};

template <class S> void expect_equal(S const& actual, S const& expected)
{
  EXPECT_TRUE(actual == expected) << "actual=" << format_scalar(actual) << " expected=" << format_scalar(expected);
}

// GTest's EXPECT_NEAR takes double. Keep subtraction, magnitude, and tolerance
// in the tested real type, and format only the diagnostic strings.
template <class S, class R> void expect_error_at_most(S const& actual, S const& expected, R const& tolerance)
{
  using std::abs;
  auto error = abs(actual - expected);
  EXPECT_TRUE(error <= tolerance) << "actual=" << format_scalar(actual) << " expected=" << format_scalar(expected)
                                  << " error=" << format_scalar(error) << " bound=" << format_scalar(tolerance);
}

#if UNI20_ENABLE_MPFR
// Transfer the native binary value exactly into a wider independent oracle.
// Decimal round trips and casts through double would contaminate the oracle.
template <class R> mpreal widen(R value)
{
  auto p = Precision::bits(512);
  if constexpr (std::same_as<R, mpreal>)
    return value.at(p);
  else
  {
    int exponent = 0;
    using std::frexp;
    R fraction = frexp(value, &exponent);
    bool const negative = fraction < R(0);
    if (negative) fraction = -fraction;
    mpreal significand(0, p);
    for (int bit = 0; bit < numeric_limits<R>::digits; ++bit)
    {
      fraction *= R(2);
      significand *= 2;
      if (fraction >= R(1))
      {
        significand += 1;
        fraction -= R(1);
      }
    }
    auto result = significand * pow(mpreal(2, p), mpreal(exponent - numeric_limits<R>::digits, p));
    return negative ? -result : result;
  }
}

template <class C> void expect_rational_accuracy(typename C::real_type component, int numerator, int denominator)
{
  auto check = Precision::bits(512);
  int const bits = C::digits();
  int const previous_bits = bits <= 24    ? 12
                            : bits <= 53  ? 24
                            : bits <= 64  ? 53
                            : bits <= 113 ? 64
                            : bits <= 128 ? 113
                                          : 128;
  auto lower = Precision::bits(previous_bits);
  mpreal const reference = mpreal(numerator, check) / mpreal(denominator, check);
  mpreal const low = mpreal(numerator, lower) / mpreal(denominator, lower);
  auto const error = abs(widen(component) - reference);
  auto const low_error = abs(low.at(check) - reference);
  EXPECT_GT(error, 0); // These non-dyadic references are not exactly representable.
  EXPECT_LE(error, widen(C::epsilon()));
  EXPECT_LT(error * 16, low_error);
}
#endif
} // namespace uni20::test
