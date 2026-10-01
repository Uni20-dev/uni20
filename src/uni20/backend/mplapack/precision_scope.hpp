#pragma once

// This header belongs only at the external-provider boundary. MPLAPACK must be
// included before its bundled wrapper so both use the same context ABI.
#include <mpblas_mpfr.h>
#include <uni20/core/precision.hpp>

namespace uni20::mplapack::detail
{
/// \brief Restore real and complex provider defaults around one synchronous invocation.
/// \warning Must never live across an await or a transfer to another thread.
class precision_scope {
  public:
    explicit precision_scope(Precision p)
        : precision_(mpfr_get_default_prec()), rounding_(mpfr_get_default_rounding_mode()), emin_(mpfr_get_emin()),
          emax_(mpfr_get_emax()), stable_rounding_(gmpfrxx_mkII::detail::stable_mpfr_rounding_mode_storage()),
          real_initialized_(gmpfrxx_mkII::detail::mpfr_defaults_initialized_storage()),
          complex_initialized_(mpfrxx::mpc_defaults_initialized_storage()),
          complex_precision_(mpfrxx::mpc_precision_override_storage()),
          complex_rounding_(mpfrxx::mpc_rounding_override_storage())
    {
      auto const bits = p.bit_count(); // Validate before mutating any provider defaults.
      mpfrxx::initialize_thread_defaults();
      mpfrxx::initialize_mpc_defaults_for_current_thread();
      // Wrapper first-use initialization may read exponent bounds from the
      // environment. Preserve Uni20's executing-thread exponent range instead.
      gmpfrxx_mkII::detail::set_mpfr_default_exponent_range(emin_, emax_);
      mpfrxx::set_default_precision_bits(bits);
      mpfrxx::set_default_rounding_mode(MPFR_RNDN);
      mpfrxx::set_default_mpc_precision_bits(bits);
      mpfrxx::set_default_mpc_rounding_mode(MPFR_RNDN);
    }
    ~precision_scope() noexcept
    {
      mpfr_set_default_prec(precision_);
      mpfr_set_default_rounding_mode(rounding_);
      gmpfrxx_mkII::detail::set_mpfr_default_exponent_range(emin_, emax_);
      gmpfrxx_mkII::detail::stable_mpfr_rounding_mode_storage() = stable_rounding_;
      gmpfrxx_mkII::detail::mpfr_defaults_initialized_storage() = real_initialized_;
      mpfrxx::mpc_defaults_initialized_storage() = complex_initialized_;
      mpfrxx::mpc_precision_override_storage() = complex_precision_;
      mpfrxx::mpc_rounding_override_storage() = complex_rounding_;
    }
    precision_scope(precision_scope const&) = delete;
    precision_scope& operator=(precision_scope const&) = delete;

  private:
    mpfr_prec_t precision_;
    mpfr_rnd_t rounding_;
    mpfr_exp_t emin_, emax_;
    mpfr_rnd_t stable_rounding_;
    bool real_initialized_, complex_initialized_;
    mpfrxx::mpc_precision_override_state complex_precision_;
    mpfrxx::mpc_rounding_override_state complex_rounding_;
};
} // namespace uni20::mplapack::detail
