#include <uni20/core/mpreal.hpp>
#include <uni20/core/mpreal_special.hpp>

#include <cstdlib>
#include <new>
#include <string_view>

enum class Operation
{
  trigonometric,
  exponential,
  log_gamma
};

// Intentional leaks are confined to subprocess negative controls.
[[gnu::noinline]] void evaluate(std::string_view mode, Operation operation)
{
  {
    auto const precision = uni20::Precision::bits(333);
    // Use the positive noninteger Stirling path exercised by numerical
    // integrands; the integer log-Gamma shortcut has separate MPFR pools.
    uni20::mpreal input(operation == Operation::log_gamma ? "3.5" : "3", precision);
    auto compute = [&]() -> uni20::mpreal {
      switch (operation)
      {
        case Operation::exponential:
          return exp(input);
        case Operation::log_gamma:
          return lgamma(input);
        case Operation::trigonometric:
          return atan(input);
      }
      std::abort();
    };
    if (mode == "owned")
    {
      auto* abandoned = new uni20::mpreal(compute());
      asm volatile("" : : "r"(abandoned) : "memory");
      // Release the wrapper without its destructor: only its MPFR payload
      // leaks. Its allocation is in Uni20's owning-result wrapper, not the
      // MPFR internal C function whose caches the policy suppresses.
      ::operator delete(static_cast<void*>(abandoned));
    }
    else
    {
      auto result = compute();
      if (operation == Operation::trigonometric) result += cos(input) + uni20::pi<uni20::mpreal>.at(precision);
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
  Operation operation = Operation::trigonometric;
  if (mode.ends_with("-exp"))
  {
    operation = Operation::exponential;
    mode.remove_suffix(4);
  }
  else if (mode.ends_with("-lgamma"))
  {
    operation = Operation::log_gamma;
    mode.remove_suffix(7);
  }
  if (mode != "cache" && mode != "clean" && mode != "owned" && mode != "plain") return 2;

  // The test driver disables TLS roots for this isolated process so the cache
  // control does not depend on which threads a scheduler happens to use.
  evaluate(mode, operation);
}
