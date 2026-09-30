#include <iostream>
#include <uni20/core/mpreal.hpp>

int main()
{
  using namespace uni20;
  using namespace uni20::literals;
  auto p = Precision::decimal_digits(80);
  mpreal deferred;
  std::cout << "Default construction is unset: " << !deferred.initialized()
            << "; assignment supplies both value and precision.\n";
  deferred = mpreal("0.1", p);
  std::cout << "Assigned value: " << deferred << "\n";
  std::cout << "Explicit arbitrary precision: " << p.bit_count() << " binary significand bits.\n"
            << "Decimal literal arithmetic is exact until .at(p) selects working precision.\n";
  auto exact = 0.1_mp + 0.2_mp;
  std::cout << "0.1_mp + 0.2_mp = " << exact << " exactly\n"
            << "At working precision: " << exact.at(p) << "\n"
            << "1/3 at working precision: " << (1_mp / 3_mp).at(p) << "\n";
  auto x = (2_mp).at(p);
  auto root = sqrt(x);
  std::cout << "sqrt(2): " << root << "\n"
            << "sqrt(2)^2 - 2 (rounding residual): " << root * root - x << "\n"
            << "pi computed at working precision: " << pi<mpreal>.at(p) << "\n"
            << "2*pi, precision inferred from x: " << x * pi<mpreal> << "\n";
  auto higher = Precision::bits(400);
  try
  {
    (void)(x + x.at(higher));
  }
  catch (std::invalid_argument const& error)
  {
    std::cout << "Mixed precision is rejected: " << error.what() << "\n";
  }
  std::cout << "Explicit promotion: " << x.at(higher) + x.at(higher) << "\n";
}
