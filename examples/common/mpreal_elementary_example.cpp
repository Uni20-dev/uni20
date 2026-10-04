#include <iostream>
#include <uni20/core/math.hpp>

// One algorithm for native and runtime precision, without a double conversion.
template <class Real> void show_small_argument(char const* label, Real const& x)
{
  auto one = uni20::scalar_like(x, 1);
  std::cout << label << ": x = " << x << "\n"
            << "  exp(x) - 1 loses the result: " << uni20::math::exp(x) - one << "\n"
            << "  expm1(x) retains it:        " << uni20::math::expm1(x) << "\n"
            << "  log(1 + x) loses it:        " << uni20::math::log(Real(one + x)) << "\n"
            << "  log1p(x) retains it:        " << uni20::math::log1p(x) << "\n";
}

int main()
{
  using namespace uni20;
  auto p = Precision::bits(256);
  std::cout << "Small arguments need dedicated elementary functions, even at high precision.\n"
            << "This example calls the same uni20::math interface for double and mpreal.\n";
  show_small_argument("binary64", 1e-100);
  show_small_argument("MPFR at 256 bits", mpreal("1e-100", p));
  std::cout << "\nRational results remain exact without choosing a working precision:\n"
            << "  cbrt(-8/27) = " << math::cbrt(mpreal("-8/27", Precision::exact())) << "\n"
            << "  exp2(-3) = " << math::exp2(mpreal{-3}) << "\n"
            << "  log2(8) = " << math::log2(mpreal{8}) << "\n"
            << "An irrational result needs precision: asinh(1) at 256 bits = "
            << math::asinh(mpreal{1}, p) << "\n";
}
