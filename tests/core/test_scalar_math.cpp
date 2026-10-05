#include <uni20/core/math.hpp>

#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

// Declared after math.hpp: the dispatcher must find these through ADL at
// instantiation, including hidden friends and an associated Uni20 namespace.
namespace scalar_math_test
{
struct Number
{
    int value;
    friend constexpr Number sqrt(Number x) noexcept { return {x.value + 100}; }
    friend constexpr Number abs(Number x) noexcept { return {x.value < 0 ? -x.value : x.value}; }
    friend constexpr Number expm1(Number x) noexcept { return {x.value + 400}; }
    friend Number exp(Number) { throw std::domain_error("custom exponential"); }
};
template <class Tag> struct TemplatedNumber
{
    int value;
};
template <class Tag> constexpr auto sqrt(TemplatedNumber<Tag> const& x) noexcept
{
  return TemplatedNumber<Tag>{x.value + 200};
}
template <class Tag> struct ConversionOnly
{
    operator double() const { return 4; }
};
struct Unsupported
{};
struct ThrowingCopy
{
    ThrowingCopy() = default;
    ThrowingCopy(ThrowingCopy const&) { throw std::runtime_error("component copy"); }
};
struct Components
{
    ThrowingCopy component;
    friend ThrowingCopy const& real(Components const& x) noexcept { return x.component; }
    friend ThrowingCopy const& imag(Components const& x) noexcept { return x.component; }
};
} // namespace scalar_math_test

namespace uni20
{
struct MathAdlTestNumber
{
    int value;
    friend constexpr MathAdlTestNumber sqrt(MathAdlTestNumber x) noexcept { return {x.value + 300}; }
};
} // namespace uni20

namespace
{
namespace m = uni20::math;
using scalar_math_test::Number;
static_assert(m::sqrt(Number{1}).value == 101);
static_assert(m::expm1(Number{1}).value == 401);

template <class... Functions> consteval bool elementary_signatures(Functions...)
{
  return ((std::invocable<Functions, float> && std::invocable<Functions, double> &&
           std::invocable<Functions, long double> &&
           !std::invocable<Functions, scalar_math_test::Unsupported> &&
           !std::invocable<Functions, scalar_math_test::ConversionOnly<std::true_type>> &&
           !std::invocable<Functions, double, double>) && ...);
}
static_assert(elementary_signatures(m::exp2, m::expm1, m::log2, m::log10, m::log1p, m::cbrt, m::asin, m::acos,
                                    m::tan, m::atan, m::sinh, m::cosh, m::tanh, m::asinh, m::acosh, m::atanh));

// Enabling MPFR must not supply missing native functions through conversion
// to mpreal. These native implementations are deliberately deferred.
template <class T> consteval bool deferred_native_special_functions()
{
  return !std::invocable<decltype(m::digamma), T> && !std::invocable<decltype(m::zeta), T> &&
         !std::invocable<decltype(m::expint), T> && !std::invocable<decltype(m::dilog_real), T> &&
         !std::invocable<decltype(m::airy_ai), T> && !std::invocable<decltype(m::beta), T, T> &&
         !std::invocable<decltype(m::upper_gamma), T, T> && !std::invocable<decltype(m::agm), T, T>;
}
static_assert(deferred_native_special_functions<float>());
static_assert(deferred_native_special_functions<double>());
static_assert(deferred_native_special_functions<long double>());
#if UNI20_HAS_FLOAT128
static_assert(deferred_native_special_functions<uni20::float128>());
#endif
static_assert(deferred_native_special_functions<scalar_math_test::ConversionOnly<std::true_type>>());
static_assert(!std::invocable<decltype(m::factorial), int>);
template <class... F> consteval bool deferred_extensions(F...)
{
  return ((!std::invocable<F, float> && !std::invocable<F, double> && !std::invocable<F, long double>
#if UNI20_HAS_FLOAT128
           && !std::invocable<F, uni20::float128>
#endif
           && !std::invocable<F, scalar_math_test::ConversionOnly<std::true_type>>) && ...);
}
static_assert(deferred_extensions(m::exp10, m::exp2m1, m::exp10m1, m::log2p1, m::log10p1,
    m::sinpi, m::cospi, m::tanpi, m::asinpi, m::acospi, m::atanpi, m::sec, m::csc, m::cot,
    m::sech, m::csch, m::coth));
static_assert(!std::invocable<decltype(m::atan2pi), double, double>);
static_assert(!std::invocable<decltype(m::compound), double, int>);


template <class T> void check_native_special_functions()
{
  static_assert(std::same_as<decltype(m::tgamma(T{5})), T>);
  static_assert(std::same_as<decltype(m::erf(T{0})), T>);
  static_assert(std::same_as<decltype(m::erfc(T{0})), T>);
  EXPECT_TRUE(m::abs(m::tgamma(T{5}) - T{24}) <= T{96} * uni20::numeric_limits<T>::epsilon());
  EXPECT_TRUE(m::erf(T{0}) == T{0});
  EXPECT_TRUE(m::erfc(T{0}) == T{1});
}

TEST(ScalarMath, NativeSpecialFunctionsKeepNativeTypes)
{
  check_native_special_functions<float>();
  check_native_special_functions<double>();
  check_native_special_functions<long double>();
#if UNI20_HAS_FLOAT128
  check_native_special_functions<uni20::float128>();
#endif
}

#if UNI20_ENABLE_MPFR
template <class... Functions> consteval bool mpfr_special_signatures(Functions...)
{
  return ((std::invocable<Functions, uni20::mpreal> &&
           std::invocable<Functions, uni20::mpreal, uni20::Precision> &&
           !std::invocable<Functions, double, uni20::Precision>) && ...);
}
static_assert(mpfr_special_signatures(m::tgamma, m::erf, m::erfc, m::lgamma, m::lgamma_sign, m::digamma,
                                      m::zeta, m::expint, m::dilog_real, m::airy_ai));
static_assert(mpfr_special_signatures(m::exp10, m::exp2m1, m::exp10m1, m::log2p1, m::log10p1,
    m::sinpi, m::cospi, m::tanpi, m::asinpi, m::acospi, m::atanpi, m::sec, m::csc, m::cot,
    m::sech, m::csch, m::coth));
static_assert(std::invocable<decltype(m::compound), uni20::mpreal, int, uni20::Precision>);
static_assert(!std::invocable<decltype(m::compound), uni20::mpreal, bool>);
static_assert(!std::invocable<decltype(m::compound), uni20::mpreal, double>);
static_assert(!std::invocable<decltype(m::bessel_j), double, uni20::mpreal>);
static_assert(!std::invocable<decltype(m::bessel_y), bool, uni20::mpreal>);
static_assert(std::invocable<decltype(m::factorial), int, uni20::Precision>);
static_assert(!std::invocable<decltype(m::factorial), bool, uni20::Precision>);
static_assert(!std::invocable<decltype(m::factorial), double, uni20::Precision>);
#endif

static_assert(std::invocable<decltype(m::atan2), double, double>);
static_assert(std::invocable<decltype(m::hypot), double, double>);
static_assert(!std::invocable<decltype(m::hypot), scalar_math_test::ConversionOnly<std::true_type>, double>);
static_assert(m::sqrt(scalar_math_test::TemplatedNumber<void>{1}).value == 201);
static_assert(m::sqrt(uni20::MathAdlTestNumber{1}).value == 301);
static_assert(noexcept(m::sqrt(Number{1})));
static_assert(!noexcept(m::exp(Number{1})));
static_assert(!std::invocable<decltype(m::sqrt), scalar_math_test::Unsupported>);
static_assert(!std::invocable<decltype(m::sqrt), scalar_math_test::ConversionOnly<void>>);
// The std template argument makes std an associated namespace. Its sqrt
// overloads must still not accept this class merely by converting to double.
static_assert(!std::invocable<decltype(m::sqrt), scalar_math_test::ConversionOnly<std::true_type>>);
static_assert(!std::invocable<decltype(m::abs), scalar_math_test::ConversionOnly<std::true_type>>);
static_assert(!std::invocable<decltype(m::sqrt)>);
static_assert(!std::invocable<decltype(m::sqrt), double, double>);
static_assert(!std::invocable<decltype(m::sqrt), double*>);
static_assert(std::same_as<decltype(m::sqrt(4)), double>);
static_assert(std::same_as<decltype(m::abs(-4)), int>);
static_assert(m::real(4) == 4 && m::imag(4) == 0 && m::conj(4) == 4);
static_assert(m::imag(1.25) == 0.0);
static_assert(!noexcept(m::real(std::declval<scalar_math_test::Components const&>())));
static_assert(!noexcept(m::imag(std::declval<scalar_math_test::Components const&>())));

TEST(ScalarMath, UsesClassCustomizationAndPreservesExceptions)
{
  EXPECT_EQ(m::abs(Number{-5}).value, 5);
  EXPECT_EQ(m::sqrt(Number{2}).value, 102);
  EXPECT_THROW(m::exp(Number{0}), std::domain_error);
  scalar_math_test::Components components;
  EXPECT_THROW(m::real(components), std::runtime_error);
  EXPECT_THROW(m::imag(components), std::runtime_error);
}

TEST(ScalarMath, ReadsComponentsByValue)
{
  uni20::complex<double> z{3, 4};
  static_assert(std::same_as<decltype(m::real(z)), double>);
  static_assert(std::same_as<decltype(m::imag(z)), double>);
  auto re = m::real(z);
  re = 9;
  EXPECT_EQ(re, 9);
  EXPECT_EQ(z.real(), 3);
  EXPECT_EQ(m::imag(z), 4);
  EXPECT_EQ(m::conj(z), (uni20::complex<double>{3, -4}));
  EXPECT_EQ(m::abs(z), 5);
  EXPECT_EQ(m::abs_squared(z), 25);
}

#if UNI20_ENABLE_MPFR
TEST(ScalarMath, ExactAndExplicitPrecisionContracts)
{
  using uni20::mpreal;
  using uni20::Precision;
  static_assert(!std::invocable<decltype(m::sqrt), double, Precision>);
  static_assert(std::invocable<decltype(m::log2), mpreal>);
  EXPECT_TRUE(m::sqrt(mpreal{4}).is_exact());
  EXPECT_EQ(m::sqrt(mpreal{4}), 2);
  EXPECT_EQ(m::pow(mpreal{2}, mpreal{-3}), mpreal("1/8", Precision::exact()));
  EXPECT_EQ(m::sqrt(uni20::exact_constant("9/4")), uni20::exact_constant("3/2"));
  EXPECT_THROW(m::sqrt(mpreal{2}), std::logic_error);
  EXPECT_THROW(m::exp(mpreal{1}), std::logic_error);
  for (auto p : {Precision::bits(80), Precision::bits(256)})
  {
    auto root = m::sqrt(mpreal{4}, p);
    EXPECT_FALSE(root.is_exact());
    EXPECT_EQ(root.precision(), p);
    EXPECT_EQ(root, 2);
    auto value = mpreal(3, p);
    EXPECT_EQ(m::real(value), value);
    EXPECT_EQ(m::real(value).precision(), p);
    EXPECT_EQ(m::imag(value), 0);
    EXPECT_EQ(m::imag(value).precision(), p);
    EXPECT_EQ(m::conj(value).precision(), p);
    EXPECT_EQ(m::exp(mpreal{}, p), 1);
    EXPECT_EQ(m::sin(mpreal{}, p), 0);
    EXPECT_EQ(m::cos(mpreal{}, p), 1);
    EXPECT_EQ(m::log(mpreal{1}, p), 0);
  }
  EXPECT_TRUE(m::imag(mpreal{3}).is_exact());
  EXPECT_THROW(m::sqrt(mpreal{2}, Precision::exact()), std::logic_error);
  EXPECT_THROW(m::sqrt(mpreal(uni20::uninitialized)), std::logic_error);
  EXPECT_THROW(m::real(mpreal(uni20::uninitialized)), std::logic_error);
  EXPECT_THROW(m::imag(mpreal(uni20::uninitialized)), std::logic_error);
}
#endif

#if UNI20_ENABLE_MPC
TEST(ScalarMath, ExactComplexAndApproximateMagnitude)
{
  using uni20::mpreal;
  using uni20::Precision;
  using C = uni20::complex<mpreal>;
  C const z(mpreal{3}, mpreal{4});
  EXPECT_EQ(m::sqrt(z), C(mpreal{2}, mpreal{1}));
  EXPECT_TRUE(m::sqrt(z).is_exact());
  EXPECT_EQ(m::abs_squared(z), 25);
  EXPECT_EQ(m::conj(z), C(mpreal{3}, mpreal{-4}));
  EXPECT_EQ(m::real(z), 3);
  EXPECT_EQ(m::imag(z), 4);
  EXPECT_THROW(m::abs(C(mpreal{1}, mpreal{1})), std::logic_error);
  auto p = Precision::bits(256);
  auto magnitude = m::abs(C(mpreal{1}, mpreal{1}), p);
  EXPECT_EQ(magnitude.precision(), p);
  EXPECT_EQ(magnitude, m::sqrt(mpreal{2}, p));
}
#endif
} // namespace
