#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "Riemann zeta retains the argument's native precision.\n"
               "At x=2 its value is pi^2/6; increasing precision reveals more digits:\n";
  std::cout << "float32: " << format_scalar(math::zeta(2.f)) << '\n';
  std::cout << "float64: " << format_scalar(math::zeta(2.)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "float80: " << format_scalar(math::zeta(float80(2))) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "float128: " << format_scalar(math::zeta(float128(2))) << '\n';
#endif
  std::cout << "\nNegative even integers are exact zeros. Their neighboring values have opposite signs:\n";
  for (double x : {math::next_down(-16.), -16., math::next_up(-16.)})
    std::cout << "zeta(" << format_scalar(x) << ") = " << format_scalar(math::zeta(x)) << '\n';
  std::cout << "\nReflection stays finite when a Gamma intermediate would overflow:\n";
  std::cout << "zeta(-35), float32: " << format_scalar(math::zeta(-35.f)) << '\n';
  std::cout << "\nThe pole at one is represented by positive infinity: " << format_scalar(math::zeta(1.)) << '\n';
  std::cout << "All calculations above use native arithmetic, without MPFR.\n";
}
