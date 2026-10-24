#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "Ei is the real principal-value exponential integral.\n"
               "It has a pole at zero and crosses zero between 1/4 and 1/2.\n"
               "Each call keeps its native type and uses Uni20 native algorithms:\n";
  std::cout << "Ei(1/4), float32: " << format_scalar(math::expint(0.25f)) << '\n';
  std::cout << "Ei(1/4), float64: " << format_scalar(math::expint(0.25)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "Ei(1/4), float80: " << format_scalar(math::expint(float80(0.25))) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "Ei(1/4), float128: " << format_scalar(math::expint(float128(0.25))) << '\n';
#endif
  std::cout << "\nOn the negative real line Ei(-x)=-E1(x) tends to negative zero.\n";
  for (double x : {-1.0, -10.0, 0.5})
    std::cout << "Ei(" << x << ") = " << format_scalar(math::expint(x)) << '\n';
#if UNI20_HAS_FLOAT128
  std::cout << "\nThe same native algorithm also supports binary128. At x=11360, exp(x) overflows\n"
               "but Ei(x) is finite; a sufficiently small negative tail remains nonzero:\n";
  std::cout << "exp(11360) = " << format_scalar(math::exp(float128(11360))) << '\n';
  std::cout << "Ei(11360) = " << format_scalar(math::expint(float128(11360))) << '\n';
  std::cout << "Ei(-11400) = " << format_scalar(math::expint(float128(-11400))) << '\n';
#endif
  std::cout << "Native calculations above do not use MPFR.\n";
}
