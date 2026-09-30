#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/tensor/tensor.hpp>

int main()
{
  using namespace uni20;
  using namespace uni20::literals;
  auto p = Precision::bits(256), low = Precision::bits(3);
  DenseMatrix<mpreal> values(2, 2, p);
  values[0, 0] = mpreal("1.125", p);
  values[1, 0] = mpreal("1.375", p);
  auto view = reshape_view(values, 4);
  values.default_precision(low);
  std::cout << "Tensor precision defaults control future initialization, independently of stored values.\n"
            << "After changing the default to 3 bits, the first element still has "
            << values[0, 0].precision().bit_count() << " bits.\n"
            << "A view created earlier keeps its " << view.default_precision().bit_count()
            << "-bit default snapshot.\n";
  auto converted = at_precision(values, low);
  std::cout << "Explicit conversion to 3 bits rounds 1.125 to " << converted[0, 0] << " and 1.375 to "
            << converted[1, 0] << " (nearest-even ties).\n"
            << "The source values remain " << values[0, 0] << " and " << values[1, 0] << ".\n";
  return converted[0, 0] == 1 && converted[1, 0] == 1.5_mp ? 0 : 1;
}
