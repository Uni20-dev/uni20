#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  double x = 720;
  std::cout << "Reciprocal hyperbolic tails in double precision, x = 720.\n"
               "cosh(x) and sinh(x) overflow, although their reciprocals remain representable.\n";
  std::cout << "1 / cosh(x) = " << format_scalar(1 / math::cosh(x)) << '\n';
  std::cout << "sech(x) = " << format_scalar(math::sech(x)) << '\n';
  std::cout << "csch(x) = " << format_scalar(math::csch(x)) << '\n';
  std::cout << "csch(-x) = " << format_scalar(math::csch(-x)) << '\n';
  std::cout << "\nAt x = 1/4 radians, reciprocal trigonometric functions use native argument reduction.\n";
  std::cout << "sec(1/4) = " << format_scalar(math::sec(0.25)) << '\n';
  std::cout << "csc(1/4) = " << format_scalar(math::csc(0.25)) << '\n';
  std::cout << "cot(1/4) = " << format_scalar(math::cot(0.25)) << '\n';
  std::cout << "\nCatalan constants retain the requested native precision:\n";
  std::cout << "float32: " << format_scalar(catalan<float>) << '\n';
  std::cout << "float64: " << format_scalar(catalan<double>) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "float80: " << format_scalar(catalan<float80>) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "float128: " << format_scalar(catalan<float128>) << '\n';
#endif
  std::cout << "No native call above uses MPFR.\n";
}
