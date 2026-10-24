#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "Airy Ai solves y''=x*y. It decays for positive x and oscillates for negative x.\n"
               "Its value at zero is 1/(cbrt(9)*Gamma(2/3)); native precision is preserved:\n";
  std::cout << "float32: " << format_scalar(math::airy_ai(0.f)) << '\n';
  std::cout << "float64: " << format_scalar(math::airy_ai(0.)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "float80: " << format_scalar(math::airy_ai(float80(0))) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "float128: " << format_scalar(math::airy_ai(float128(0))) << '\n';
#endif
  std::cout << "\nSome points on the oscillatory and decaying sides:\n";
  for (double x : {-10., -3., 0., 3., 10.})
    std::cout << "Ai(" << x << ") = " << format_scalar(math::airy_ai(x)) << '\n';
  std::cout << "\nAt a huge negative argument, the phase is reduced from the original binary input.\n"
               "A native x*sqrt(-x) intermediate would overflow here:\n"
            << "Ai(-1e300) = " << format_scalar(math::airy_ai(-1e300)) << '\n';
  std::cout << "All calculations above use native arithmetic, without MPFR.\n";
}
