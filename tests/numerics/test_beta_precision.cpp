#include "precision_registry.hpp"
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, BetaIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto a = C::real(1) / 2, b = C::real(2);
  auto result = math::beta(a, b);
  C::expect_precision(result);
  expect_error_at_most(result, C::real(4) / 3, C::epsilon() * 64);
  expect_error_at_most(math::beta(b, a), result, C::epsilon() * 64);
}
#if UNI20_HAS_MPREAL
namespace
{
// mpfr_beta itself can overflow internally for large positive parameters.
// An independently evaluated log-Gamma identity uses enough extra bits to
// survive cancellation of terms as large as b*log(b), then rounds only once.
mpreal beta_reference(mpreal const& a, mpreal const& b)
{
  auto p = a.precision();
  if (mpfr_get_exp(a.native_handle()) >= -256 && mpfr_get_exp(b.native_handle()) >= -256 && a <= 65536 && b <= 65536)
    return math::beta(a, b);
  auto extra = std::max<mpfr_exp_t>({0, mpfr_get_exp(a.native_handle()), mpfr_get_exp(b.native_handle())}) + 64;
  auto wp = Precision::bits(mpfr_get_prec(a.native_handle()) + extra);
  auto x = a.at(wp), y = b.at(wp);
  auto xp = Precision::bits(p.bit_count() + std::max<mpfr_exp_t>(0, mpfr_get_exp(a.native_handle())) + 64);
  auto yp = Precision::bits(p.bit_count() + std::max<mpfr_exp_t>(0, mpfr_get_exp(b.native_handle())) + 64);
  auto loga = math::lgamma(a.at(xp)).at(wp), logb = math::lgamma(b.at(yp)).at(wp);
  return math::exp(loga + logb - math::lgamma(x + y)).at(p);
}
template <class C, bool Portable> void beta_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  auto one = mpreal(1, p);
  auto evaluate = [](R const& a, R const& b) {
    if constexpr (Portable)
      return detail::native_math::portable_beta(a, b);
    else
      return math::beta(a, b);
  };
  std::vector<std::pair<R, R>> inputs;
  auto add = [&](R a, R b) {
    if (a > 0 && b > 0 && math::isfinite(a) && math::isfinite(b)) inputs.emplace_back(a, b);
  };
  for (int i = 1; i <= 32; ++i)
    for (int j = 1; j <= 16; ++j)
      add(C::real(i) / 7, C::real(j) / 9);
  for (int n : {1, 8, 16, 32})
  {
    R lo = C::real(n), hi = lo;
    for (int k = 0; k < 16; ++k)
    {
      for (int d : {2, 3, 7})
      {
        add(C::real(1) / d, lo);
        add(C::real(1) / d, hi);
        add(lo, hi);
      }
      lo = math::next_down(lo);
      hi = math::next_up(hi);
    }
  }
  // The initial shifted-Gamma formula lost too much accuracy for these inputs.
  add(C::real(108) / 7, C::real(2) / 9);
  add(C::real(111) / 7, C::real(5) / 9);
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    if constexpr (std::same_as<R, float>) add(0x1.14326cp-127f, 0x1.dd1c78p-128f);
    // Locate the normal/subnormal and zero-rounding transitions independently.
    // Exercise both a huge fixed second parameter and the symmetric diagonal.
    for (bool symmetric : {false, true})
      for (bool zero_boundary : {false, true})
      {
        auto target = zero_boundary ? widen(L::denorm_min()) / 2 : widen(L::min());
        auto lo = one / 2, hi = mpreal(symmetric ? L::max_exponent : 2, p);
        for (int k = 0; k < C::digits() + 32; ++k)
        {
          auto mid = (lo + hi) / 2;
          auto reference = beta_reference(mid, symmetric ? mid : widen(L::max()));
          if (reference > target)
            lo = mid;
          else
            hi = mid;
        }
        R up = scalar_at_precision<R>((lo + hi) / 2, C::precision()), down = up;
        for (int k = 0; k < 16; ++k)
        {
          add(up, symmetric ? up : L::max());
          add(down, symmetric ? down : L::max());
          up = math::next_up(up);
          down = math::next_down(down);
        }
      }
    for (R a : {L::denorm_min(), L::min(), R(1) / L::max(), C::epsilon(), R(.125), R(.5), R(1), R(2), R(16), L::max()})
    {
      for (R b : {L::min(), R(1), R(8), R(16), R(32), L::max()})
        add(a, b);
      for (int e = L::min_exponent - L::digits(); e < L::max_exponent; e += 2048)
        add(a, math::ldexp(R(1), e));
    }
    R tiny = R(1) / L::max();
    for (int k = 0; k < 16; ++k)
    {
      add(tiny, R(16));
      tiny = math::next_up(tiny);
    }
    std::uint64_t state = 0x42657461ULL;
    auto next = [&]() {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      return state;
    };
    auto overflow_midpoint = widen(L::max()) + math::ldexp(one, L::max_exponent - C::digits() - 1);
    for (int k = 0; k < 64; ++k)
    {
      R fraction = math::ldexp(R(next()), -64) + math::ldexp(R(next()), -128);
      R a = L::min() * (R(.25) + R(.75) * fraction);
      auto difference = overflow_midpoint - one / widen(a);
      if (difference <= 0) continue;
      R b = scalar_at_precision<R>(one / difference, C::precision());
      add(a, math::next_down(b));
      add(a, b);
      add(a, math::next_up(b));
    }
    for (int k = 0; k < 128; ++k)
    {
      R fraction = math::ldexp(R(next()), -64) + math::ldexp(R(next()), -128);
      int e =
          static_cast<int>(next() % (L::max_exponent - L::min_exponent + L::digits())) + L::min_exponent - L::digits();
      add(R(next() % 10000 + 1) / 1000, math::ldexp(fraction, e));
    }
  }
  auto maximum = mpreal(0, p), worsta = one, worstb = one;
  auto epsilon = math::ldexp(one, 1 - C::digits());
  auto half_denorm = mpreal(0, p);
  if constexpr (!C::runtime) half_denorm = widen(numeric_limits<R>::denorm_min()) / 2;
  auto check = [&](R const& a, R const& b, mpreal const& reference) {
    SCOPED_TRACE(format_scalar(a) + "," + format_scalar(b));
    auto result = evaluate(a, b);
    C::expect_precision(result);
    ASSERT_FALSE(math::isnan(reference));
    R rounded = scalar_at_precision<R>(reference, C::precision());
    if (math::isinf(rounded))
    {
      EXPECT_EQ(result, rounded);
      return;
    }
    ASSERT_TRUE(math::isfinite(result));
    if (reference == 0)
    {
      EXPECT_EQ(result, 0);
      return;
    }
    auto bound = 32 * epsilon * std::max(one, abs(math::log(reference))) * reference + half_denorm;
    auto error = abs(widen(result) - reference) / bound;
    EXPECT_LE(error, one);
    if (error > maximum)
    {
      maximum = error;
      worsta = widen(a);
      worstb = widen(b);
    }
    if constexpr (Portable)
    {
      EXPECT_EQ(result, evaluate(b, a));
      if constexpr (!requires { detail::native_math::standard_special_provider<R>::beta(a, b); })
      {
        EXPECT_EQ(result, math::beta(a, b));
      }
    }
  };
  for (auto const& [a, b] : inputs)
    check(a, b, beta_reference(widen(a), widen(b)));
  // Independent finite products for integer first parameters.
  for (int n : {1, 2, 7})
    for (int d : {2, 3, 17})
    {
      R b = C::real(d) / 2;
      auto reference = one / widen(b);
      for (int k = 1; k < n; ++k)
        reference *= k / (widen(b) + k);
      check(C::real(n), b, reference);
    }
  testing::Test::RecordProperty("beta_max_bound_fraction", format_scalar(maximum));
  testing::Test::RecordProperty("beta_worst_a", format_scalar(worsta));
  testing::Test::RecordProperty("beta_worst_b", format_scalar(worstb));
  EXPECT_EQ(beta_reference(worsta, worstb),
            beta_reference(worsta.at(Precision::bits(1024)), worstb.at(Precision::bits(1024))).at(p));
  auto reference = pi<mpreal>.at(p);
  auto error = abs(widen(evaluate(C::real(1) / 2, C::real(1) / 2)) - reference);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto previous_error = abs(reference.at(Precision::bits(previous)).at(p) - reference);
  EXPECT_LT(error * 16, previous_error);
}
} // namespace
UNI20_PRECISION_TEST(NumericalScalar, BetaAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  beta_accuracy<TypeParam, false>();
}
UNI20_PRECISION_TEST(NumericalScalar, BetaPortableAccuracy)
{
  this->RecordProperty("backend", "portable_native_math");
  beta_accuracy<TypeParam, true>();
}
#endif
} // namespace uni20::test
