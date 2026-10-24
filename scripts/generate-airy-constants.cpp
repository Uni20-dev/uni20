// Offline coefficient generator. Build with a C++ compiler and -lmpfr -lgmp.
// MPFR is needed to regenerate constants, never to evaluate native Airy.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mpfr.h>
void print(char const* name, mpfr_srcptr x)
{
  mpfr_t v;
  mpfr_init2(v, 2048);
  mpfr_abs(v, x, MPFR_RNDN);
  printf("inline constexpr std::uint16_t %s[] = {\n", name);
  for (int k = 0; k < 32; ++k)
  {
    mpfr_mul_2ui(v, v, 16, MPFR_RNDN);
    auto chunk = mpfr_get_ui(v, MPFR_RNDD);
    mpfr_sub_ui(v, v, chunk, MPFR_RNDN);
    printf("0x%04lx,%s", chunk, k % 8 == 7 ? "\n" : " ");
  }
  puts("};");
  mpfr_clear(v);
}
int main()
{
  mpfr_t z, a, b, c, h, plus, minus, check;
  mpfr_inits2(2048, z, a, b, c, h, plus, minus, check, (mpfr_ptr)0);
  mpfr_set_zero(z, 1);
  mpfr_ai(a, z, MPFR_RNDN);
  mpfr_set_ui(b, 1, MPFR_RNDN);
  mpfr_div_ui(b, b, 3, MPFR_RNDN);
  mpfr_gamma(b, b, MPFR_RNDN);
  mpfr_set_ui(c, 3, MPFR_RNDN);
  mpfr_rootn_ui(c, c, 3, MPFR_RNDN);
  mpfr_mul(b, b, c, MPFR_RNDN);
  mpfr_ui_div(b, 1, b, MPFR_RNDN);
  mpfr_neg(b, b, MPFR_RNDN);
  mpfr_set_ui(h, 1, MPFR_RNDN);
  mpfr_div_2ui(h, h, 400, MPFR_RNDN);
  mpfr_ai(plus, h, MPFR_RNDN);
  mpfr_neg(minus, h, MPFR_RNDN);
  mpfr_ai(minus, minus, MPFR_RNDN);
  mpfr_sub(check, plus, minus, MPFR_RNDN);
  mpfr_div(check, check, h, MPFR_RNDN);
  mpfr_div_ui(check, check, 2, MPFR_RNDN);
  mpfr_sub(check, check, b, MPFR_RNDN);
  mpfr_abs(check, check, MPFR_RNDN);
  mpfr_log2(check, check, MPFR_RNDN);
  mpfr_fprintf(stderr, "derivative independent difference log2=%.8Rf\n", check);
  if (mpfr_cmp_si(check, -700) > 0) return EXIT_FAILURE;
  // Ai(0) also has an independent Gamma identity, evaluated far beyond the
  // 512 stored coefficient bits.
  mpfr_set_ui(check, 2, MPFR_RNDN);
  mpfr_div_ui(check, check, 3, MPFR_RNDN);
  mpfr_gamma(check, check, MPFR_RNDN);
  mpfr_set_ui(c, 9, MPFR_RNDN);
  mpfr_rootn_ui(c, c, 3, MPFR_RNDN);
  mpfr_mul(check, check, c, MPFR_RNDN);
  mpfr_ui_div(check, 1, check, MPFR_RNDN);
  mpfr_sub(check, check, a, MPFR_RNDN);
  mpfr_abs(check, check, MPFR_RNDN);
  mpfr_log2(check, check, MPFR_RNDN);
  mpfr_fprintf(stderr, "Ai(0) independent difference log2=%.8Rf\n", check);
  if (mpfr_cmp_si(check, -1800) > 0) return EXIT_FAILURE;
  puts("// Generated coefficients; see scripts/generate-airy-phase.py and\n// scripts/generate-airy-constants.cpp, and "
       "docs/tensor/native_airy.md.\n#pragma once\n#include <cstdint>\nnamespace "
       "uni20::detail::native_math::airy_detail {");
  print("ai_zero_bits", a);
  print("ai_prime_zero_bits", b);
  mpfr_const_log2(c, MPFR_RNDN);
  print("log_two_bits", c);
  mpfr_const_pi(c, MPFR_RNDN);
  mpfr_log(c, c, MPFR_RNDN);
  mpfr_sub_ui(c, c, 1, MPFR_RNDN);
  print("log_pi_fraction_bits", c);
  puts("}");
  mpfr_clears(z, a, b, c, h, plus, minus, check, (mpfr_ptr)0);
}
