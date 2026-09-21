#include <uni20/common/aligned_buffer.hpp>

#include <cstring>

// Keep expected failures outside the unit-test process and avoid uninstrumented C++ libraries.
int main(int argc, char** argv)
{
  if (argc != 2) return 2;
  auto values = uni20::allocate_uninitialized_buffer<double>(2);
  values[0] = 3;
  if (std::strcmp(argv[1], "complete") == 0) values[1] = 4;
  if (std::strcmp(argv[1], "copy") == 0)
  {
    auto copy = uni20::allocate_uninitialized_buffer<double>(2);
    std::memcpy(copy.get(), values.get(), 2 * sizeof(double));
    uni20::detail::memory_diagnostics::check_initialized(copy.get(), 2 * sizeof(double));
    return copy[1] > 0 ? 0 : 1;
  }
  uni20::detail::memory_diagnostics::check_initialized(values.get(), 2 * sizeof(double));
  return values[0] + values[1] == 7 ? 0 : 1;
}
