#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "The real dilogarithm is the series sum x^k/k^2 near zero.\n"
               "At x=1/2 it equals pi^2/12 - log(2)^2/2. Each call keeps its native type:\n";
  std::cout << "float32: " << format_scalar(math::dilog_real(0.5f)) << '\n';
  std::cout << "float64: " << format_scalar(math::dilog_real(0.5)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "float80: " << format_scalar(math::dilog_real(float80(0.5))) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "float128: " << format_scalar(math::dilog_real(float128(0.5))) << '\n';
#endif
  std::cout << "\nFor x>1 this API returns only the real part of Li2, not its imaginary branch value.\n"
               "It is positive at x=2, crosses zero between 12 and 13, and becomes negative:\n";
  for (double x : {2.0, 12.0, 13.0})
    std::cout << "Re Li2(" << x << ") = " << format_scalar(math::dilog_real(x)) << '\n';
  std::cout << "\nEvery finite native input has a finite result, including both extreme signs:\n";
  std::cout << "Re Li2(maximum double) = " << format_scalar(math::dilog_real(numeric_limits<double>::max())) << '\n';
  std::cout << "Li2(-maximum double) = " << format_scalar(math::dilog_real(-numeric_limits<double>::max())) << '\n';
  std::cout << "Native calculations above do not use MPFR.\n";
}
