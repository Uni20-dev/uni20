#include "precision_registry.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, CompoundIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto one = C::real(1), x = one / 8 + C::gap();
  auto result = math::compound(x, 3);
  C::expect_precision(result);
  // This identity uses a polynomial, independently of integer exponentiation.
  expect_error_at_most(result, one + 3 * x + 3 * x * x + x * x * x, C::epsilon() * 8);
  auto tiny = math::ldexp(one, -60);
  auto grown = math::compound(tiny, std::int64_t(1) << 60);
  C::expect_precision(grown);
  EXPECT_GT(grown, C::real(27) / 10);
  EXPECT_LT(grown, C::real(28) / 10);
}

#if UNI20_HAS_MPREAL
namespace
{
struct PublicCompound
{
    template <class R> static R evaluate(R const& x, std::int64_t n) { return math::compound(x, n); }
};
struct PortableCompound
{
    template <class R> static R evaluate(R x, std::int64_t n) { return detail::native_math::portable_compound(x, n); }
};

template <class C> mpreal compound_error(typename C::real_type const& result, mpreal const& reference)
{
  using R = typename C::real_type;
  auto p = reference.precision();
  C::expect_precision(result);
  R rounded = scalar_at_precision<R>(reference, C::precision());
  if (math::isinf(rounded))
  {
    EXPECT_EQ(result, rounded);
    return mpreal(0, p);
  }
  EXPECT_TRUE(math::isfinite(result));
  EXPECT_FALSE(math::signbit(result));
  if (!math::isfinite(result)) return mpreal(100, p);
  if (rounded == 0)
  {
    EXPECT_EQ(result, 0);
  }
  if (reference == 0) return abs(widen(result));
  int exponent = static_cast<int>(math::frexp(reference).exponent);
  if constexpr (!C::runtime) exponent = std::max(exponent, numeric_limits<R>::min_exponent);
  auto spacing = math::ldexp(mpreal(1, p), exponent - C::digits());
  auto error = abs(widen(result) - reference) / spacing;
  EXPECT_LE(error, 4);
  return error;
}

template <class C, class F> void compound_accuracy()
{
  using R = typename C::real_type;
  auto one = C::real(1);
  constexpr auto low = numeric_limits<std::int64_t>::min(), high = numeric_limits<std::int64_t>::max();
  constexpr std::array<std::int64_t, 15> powers{
      low, -high, -(std::int64_t(1) << 53) - 1, -4097, -127, -3, -1, 0, 1, 2, 3,
      127, 4097,  (std::int64_t(1) << 53) + 1,  high};
  std::vector<std::pair<R, std::int64_t>> inputs;
  for (int k = 1; k < 160; k += 3)
    for (auto n : powers)
    {
      inputs.emplace_back(math::ldexp(one, -k), n);
      inputs.emplace_back(-math::ldexp(one, -k), n);
    }
  for (int i = -63; i <= 256; ++i)
    for (auto n : powers)
      inputs.emplace_back(C::real(i) / 64, n);
  for (R center : {C::real(-1), -one / 2, C::real(0), one / 2, one})
  {
    auto up = center, down = center;
    for (int i = 0; i < 8; ++i)
    {
      for (auto n : powers)
      {
        inputs.emplace_back(up, n);
        inputs.emplace_back(down, n);
      }
      up = math::next_up(up);
      down = math::next_down(down);
    }
  }
  auto p = Precision::bits(512);
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    // Derive boundary inputs from a wider logarithm, then visit both sides.
    // No rounded native logarithm is used to decide the expected result.
    auto ln2 = math::log(mpreal(2, p));
    for (auto n : powers)
      if (n != 0)
        for (int e : {L::min_exponent - C::digits() - 1, L::min_exponent - C::digits(), L::min_exponent,
                      L::max_exponent - 1, L::max_exponent})
        {
          auto boundary = math::expm1(ln2 * e / n);
          R up = scalar_at_precision<R>(boundary, C::precision()), down = up;
          if (!math::isfinite(up)) continue;
          for (int i = 0; i < 8; ++i)
          {
            inputs.emplace_back(up, n);
            inputs.emplace_back(down, n);
            up = math::next_up(up);
            down = math::next_down(down);
          }
        }
  }
  std::uint64_t state = 0x436f6d706f756e64ULL;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (int i = 0; i < 512; ++i)
  {
    auto fraction = math::ldexp(scalar_at_precision<R>(next(), C::precision()), -64) +
                    math::ldexp(scalar_at_precision<R>(next(), C::precision()), -128);
    auto x = math::ldexp(fraction, static_cast<int>(next() % 150) - 128);
    if (next() & 1) x = -x;
    auto n = static_cast<std::int64_t>(next() >> 1);
    if (next() & 1) n = -n;
    inputs.emplace_back(x, n);
  }
  // Keep nontrivial finite outputs while varying every relevant exponent width.
  // Unconstrained large exponents otherwise mostly test overflow/underflow.
  for (int width : {1, 3, 7, 16, 23, 24, 31, 52, 53, 62, 63})
  {
    auto magnitude = static_cast<std::int64_t>((std::uint64_t(1) << width) - 1);
    for (auto n : {magnitude, -magnitude})
      for (auto target : {mpreal(-20, p), mpreal(1, p) / 4, mpreal(3, p), mpreal(20, p)})
        inputs.emplace_back(scalar_at_precision<R>(math::expm1(target / n), C::precision()), n);
  }
  // Fixed regression for the observed subnormal double rounding maximum.
  inputs.emplace_back(scalar_at_precision<R>(-0.4999999105930328369140625L, C::precision()), 127);
  // Native exp(n*log1p(x)) exceeded 100 to 20,000 spacing units on these
  // ordinary finite-output cases in the prototype scan.
  for (auto [text, n] : std::array<std::pair<char const*, std::int64_t>, 4>{
           {{"-0.0214227549731731414794921875", -4097},
            {"7.87247562375859278581352868316043792619928973e-14", 9007199254740993LL},
            {"0.308728648249661915701960029911887772868794855", -22305},
            {"-0.344237943163331067551116626908197975492820445", -16077}}})
    inputs.emplace_back(scalar_at_precision<R>(mpreal(text, p), C::precision()), n);
  auto maximum = mpreal(0, p), worst = mpreal(0, p);
  std::int64_t worst_n = 0;
  int sample = 0;
  for (auto const& [x, n] : inputs)
  {
    if (x <= -one) continue; // Core tests cover endpoints and invalid domain.
    SCOPED_TRACE(format_scalar(x));
    SCOPED_TRACE(n);
    auto wide = widen(x), reference = math::compound(wide, n);
    // MPFR's finite exponent range also overflows on astronomical powers.
    auto error = compound_error<C>(F::evaluate(x, n), reference);
    if (error > maximum)
    {
      maximum = error;
      worst = wide;
      worst_n = n;
    }
    if (++sample % 257 == 0)
    {
      auto reference_1024 = math::compound(wide.at(Precision::bits(1024)), n).at(p);
      EXPECT_EQ(reference, reference_1024);
    }
  }
  testing::Test::RecordProperty("compound_max_grid_error", format_scalar(maximum));
  testing::Test::RecordProperty("compound_worst_input", format_scalar(worst));
  testing::Test::RecordProperty("compound_worst_exponent", std::to_string(worst_n));

  // Independent exact rational reference: (1 + 3/16)^(-13) = (16/19)^13.
  auto reference = math::pown(mpreal(16) / 19, 13).at(p);
  auto result = F::evaluate(C::real(3) / 16, -13);
  compound_error<C>(result, reference);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto error = abs(widen(result) - reference);
  auto previous_error = abs(reference.at(Precision::bits(previous)).at(p) - reference);
  EXPECT_LT(error * 16, previous_error);
}
} // namespace

UNI20_PRECISION_TEST(NumericalScalar, CompoundAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  compound_accuracy<TypeParam, PublicCompound>();
}
UNI20_PRECISION_TEST(NumericalScalar, CompoundPortableAccuracy)
{
  this->RecordProperty("backend", "portable_native_math");
  compound_accuracy<TypeParam, PortableCompound>();
}
#endif
} // namespace uni20::test
