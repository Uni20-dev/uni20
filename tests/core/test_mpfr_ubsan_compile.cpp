#include <uni20/core/mpcomplex.hpp>

// Deliberately leave these external functions undefined: definitions in this
// translation unit can hide GCC's UBSan constant-expression failure.
namespace uni20::test::function_identity
{
int first(int);
int second(int);

static_assert(detail::same_function_v<first, first>);
static_assert(detail::same_function_v<&first, first>);
static_assert(!detail::same_function_v<first, second>);
static_assert(detail::same_function_v<mpfr_add, mpfr_add>);
static_assert(!detail::same_function_v<mpfr_add, mpfr_sub>);
static_assert(!detail::same_function_v<mpc_add, mpc_sub>);

// Unlike the ordinary overloads instantiated by including the headers, these
// integer-order wrappers require an explicit use to compile both dispatches.
void instantiate_integer_bessel(mpreal const& x)
{
  (void)bessel_j(1, x);
  (void)bessel_y(1, x);
}
} // namespace uni20::test::function_identity
