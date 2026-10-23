#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "Digamma is the logarithmic derivative of Gamma.\n"
               "At x=1/2 it equals -EulerGamma - 2*log(2). Each call retains its native type:\n";
  std::cout << "float32: " << format_scalar(math::digamma(0.5f)) << '\n';
  std::cout << "float64: " << format_scalar(math::digamma(0.5)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "float80: " << format_scalar(math::digamma(float80(0.5))) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "float128: " << format_scalar(math::digamma(float128(0.5))) << '\n';
#endif
  std::cout << "\nNegative nonintegers use reflection. Integer poles give NaN; neighbors have opposite signs:\n";
  std::cout << "psi(-1/2) = " << format_scalar(math::digamma(-0.5)) << '\n';
  std::cout << "psi(-1) = " << format_scalar(math::digamma(-1.0)) << '\n';
  std::cout << "psi(next_down(-1)) = " << format_scalar(math::digamma(math::next_down(-1.0))) << '\n';
  std::cout << "psi(next_up(-1)) = " << format_scalar(math::digamma(math::next_up(-1.0))) << '\n';
  std::cout << "\nFor large positive x, digamma approaches log(x) and stays finite:\n";
  std::cout << "psi(maximum double) = " << format_scalar(math::digamma(numeric_limits<double>::max())) << '\n';
  std::cout << "Native calculations above do not use MPFR.\n";
}
