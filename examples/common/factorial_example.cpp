#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  std::cout << "Typed factorials: the integer argument does not determine the floating result type.\n"
               "The native factory uses compensated multiplication and bounded overflow handling.\n";
  std::cout << "30!, float32 = " << format_scalar(factorial<float>(30)) << '\n';
  std::cout << "30!, float64 = " << format_scalar(factorial<double>(30)) << '\n';
#if UNI20_HAS_FLOAT80
  std::cout << "30!, float80 = " << format_scalar(factorial<float80>(30)) << '\n';
#endif
#if UNI20_HAS_FLOAT128
  std::cout << "30!, float128 = " << format_scalar(factorial<float128>(30)) << '\n';
#endif
  std::cout << "\nAn explicit precision is useful in generic code and must match a native result:\n";
  std::cout << "30!, double with precision_of(1.0) = " << format_scalar(factorial<double>(30, precision_of(1.0)))
            << '\n';
  std::cout << "\n171! exceeds double's range: " << format_scalar(factorial<double>(171)) << '\n';
#if UNI20_HAS_MPREAL
  std::cout << "\nMPFR uses the same typed factory with an explicit finite or exact precision:\n";
  std::cout << "30!, 160 bits = " << format_scalar(factorial<mpreal>(30, Precision::bits(160))) << '\n';
  std::cout << "30!, exact integer = " << format_scalar(factorial<mpreal>(30, Precision::exact())) << '\n';
#endif
}
