#include <uni20/core/mpreal.hpp>

#include <cstdlib>
#include <new>
#include <string_view>

// Intentional leaks are confined to subprocess negative controls.
[[gnu::noinline]] void evaluate(std::string_view mode, bool exponential)
{
  {
    auto const precision = uni20::Precision::bits(333);
    uni20::mpreal input(3, precision);
    if (mode == "owned")
    {
      auto* abandoned = new uni20::mpreal(exponential ? exp(input) : atan(input));
      asm volatile("" : : "r"(abandoned) : "memory");
      // Release the wrapper without its destructor: only its MPFR payload
      // leaks. Its allocation stack has apply<&mpfr_atan> or apply<&mpfr_exp>,
      // not the C function whose caches the policy suppresses.
      ::operator delete(static_cast<void*>(abandoned));
    }
    else
    {
      auto result = exponential ? exp(input) : atan(input) + cos(input) + uni20::pi<uni20::mpreal>.at(precision);
      if (mpfr_number_p(result.native_handle()) == 0) std::abort();
    }
    if (mode == "plain")
    {
      void* abandoned = std::malloc(97);
      asm volatile("" : : "r"(abandoned) : "memory");
    }
  }
  if (mode != "cache") mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);
}

int main(int argc, char** argv)
{
  if (argc != 2) return 2;
  std::string_view mode(argv[1]);
  bool const exponential = mode.ends_with("-exp");
  if (exponential) mode.remove_suffix(4);
  if (mode != "cache" && mode != "clean" && mode != "owned" && mode != "plain") return 2;

  // The test driver disables TLS roots for this isolated process so the cache
  // control does not depend on which threads a scheduler happens to use.
  evaluate(mode, exponential);
}
