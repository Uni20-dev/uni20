#include <iostream>
#include <uni20/core/math.hpp>
#include <vector>

// These algorithms use ordinary generic numerical construction.
template <class T> T sum(std::vector<T> const& values)
{
  T result{};
  for (auto const& x : values)
    result += x;
  return result;
}
template <class T> T product(std::vector<T> const& values)
{
  T result{1};
  for (auto const& x : values)
    result *= x;
  return result;
}
int main()
{
  using namespace uni20;
  using namespace uni20::literals;
  auto p = Precision::bits(256);
  mpreal third = mpreal{1} / mpreal{3};
  auto exact_sum = sum(std::vector<mpreal>{third, third, third});
  auto exact_product = product(std::vector<mpreal>{third, mpreal{3}});
  std::cout << "Exact rationals let ordinary T{} and T{1} accumulators work without a precision setting.\n"
            << "Three exact thirds sum to " << exact_sum << "; multiplying a third by three gives " << exact_product
            << ".\nThe same generic sum for doubles gives " << sum(std::vector<double>{0.25, 0.75}) << ".\n";
  auto mixed = sum(std::vector<mpreal>{third, mpreal(0.1_mp, p)});
  std::cout << "An approximate operand selects " << mixed.precision().bit_count() << " bits: " << mixed << '\n';
  std::cout << "An irrational square root needs finite precision; pass the working precision explicitly:\n"
            << "sqrt(2) = " << sqrt(mpreal{2}, p) << '\n';
  mpreal fraction("4/9", Precision::exact());
  auto root = sqrt(fraction);
  std::cout << "Parsing 4/9 and taking its square root stays exact: " << root << " (is_exact = " << is_exact(root)
            << ").\n"
            << "An explicit cast rounds it to double: " << static_cast<double>(root) << ".\n";
  auto approximate_root = sqrt(fraction, p);
  std::cout << "Passing precision explicitly also makes a rational root approximate: " << approximate_root
            << " (is_exact = " << is_exact(approximate_root) << ").\n";
  if (root * root != fraction || !is_exact(root) || is_exact(approximate_root)) return 1;
#if UNI20_ENABLE_MPC
  complex<mpreal> z(mpreal{1}, mpreal{2});
  auto quotient = z / complex<mpreal>{3};
  std::cout << "Complex rational arithmetic stays exact too: (1+2i)/3 = " << quotient
            << "; squared norm = " << norm(quotient) << ".\n";
  std::cout << "An irrational complex magnitude can request precision too: |1+2i| = " << abs(z, p) << ".\n";
  if (!quotient.is_exact() || quotient * complex<mpreal>{3} != z) return 1;
#endif
  return exact_sum.is_exact() && exact_sum == 1 && exact_product == 1 && mixed.precision() == p ? 0 : 1;
}
