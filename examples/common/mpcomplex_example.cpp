#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

int main()
{
  using namespace uni20;
  auto p = Precision::decimal_digits(80);
  complex<mpreal> z;
  std::cout << "MPC complex scalars: unset storage is filled by explicit-precision assignment.\n";
  z = complex<mpreal>{"3", "4", p};
  std::cout << "z = " << format_scalar(z) << " at " << z.precision().bit_count() << " bits per component\n"
            << "|z| = " << abs(z) << " (expected 5), norm = " << norm(z) << " (expected 25)\n"
            << "z * conj(z) = " << format_scalar(z * conj(z)) << " (expected 25+0i)\n";
  auto root = sqrt(complex<mpreal>{"-4", "-0", p});
  std::cout << "Principal sqrt(-4-0i) = " << format_scalar(root)
            << "; negative imaginary zero selects the lower side of the branch cut.\n";
  auto component = z.real();
  component = mpreal(99, p);
  std::cout << "Editing an extracted real component leaves z unchanged: " << format_scalar(z) << "\n";
  z.imag(mpreal("0.1", Precision::bits(400)));
  std::cout << "An imaginary-component setter rounds to z's existing precision: " << format_scalar(z) << "\n";
}
