#include <uni20/common/terminal.hpp>
#include <uni20/core/math.hpp>
#if UNI20_HAS_FLOAT128
#include <mpblas_binary128.h>
#endif

// Static call operators require C++23; the parent deliberately requests C++17.
static_assert([]() static { return true; }());
static_assert(sizeof(uni20::blas_int) == (UNI20_ILP64 ? 8 : 4));

int main()
{
  // Exercise an out-of-line common-library function, not just its headers.
  if (terminal::is_a_terminal(nullptr))
    return 1;

#if UNI20_HAS_FLOAT128
  static_assert(uni20::numeric_limits<uni20::float128>::digits == 113);
  using std::abs;
  using std::atan;
  using std::tan;
  uni20::float128 const x = uni20::float128{1} / uni20::float128{3};
  auto const error = abs(tan(atan(x)) - x);
  if (error > uni20::float128{8} * uni20::numeric_limits<uni20::float128>::epsilon())
    return 2;
  uni20::float128 values[] = {1, 2};
  if (Rdot(2, values, 1, values, 1) != uni20::float128{5})
    return 3;
#endif
  return 0;
}
