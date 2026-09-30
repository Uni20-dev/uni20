#pragma once

#include "mpreal.hpp"
#if !UNI20_ENABLE_MPC
#error "uni20/core/mpcomplex.hpp requires UNI20_ENABLE_MPC"
#endif
#include <mpc.h>

namespace uni20
{
namespace detail
{
struct mpcomplex_access;
}

/// \brief Owning MPC value, exposed as uni20::complex<mpreal>.
/// \details Both components share explicit precision. Arithmetic is eager and
///          rounds each component to nearest, ties to even. Default construction
///          is unset; numerical use of an unset value throws std::logic_error.
class mpcomplex {
  public:
    using value_type = mpreal;

    mpcomplex() noexcept = default;
    /// \brief Construct complex zero at the requested component precision.
    explicit mpcomplex(Precision p)
    {
      detail::require_mpfr_tls();
      mpc_init2(value_, p.bit_count());
      mpc_set_ui(value_, 0, MPC_RNDNN);
      initialized_ = true;
    }
    /// \brief Construct from components, explicitly converting both to p.
    mpcomplex(mpreal const& re, mpreal const& im, Precision p) : mpcomplex(p)
    {
      mpc_set_fr_fr(value_, re.native_handle(), im.native_handle(), MPC_RNDNN);
    }
    /// \brief Construct from components of matching precision.
    mpcomplex(mpreal const& re, mpreal const& im) : mpcomplex(re, im, matching_precision(re, im)) {}
    /// \brief Embed a real value, retaining its precision and adding positive imaginary zero.
    mpcomplex(mpreal const& re) : mpcomplex(re.precision()) { mpc_set_fr(value_, re.native_handle(), MPC_RNDNN); }
    mpcomplex(mpreal const& re, Precision p) : mpcomplex(p) { mpc_set_fr(value_, re.native_handle(), MPC_RNDNN); }
    /// \brief Parse two real decimal components at explicit precision.
    mpcomplex(std::string_view re, std::string_view im, Precision p) : mpcomplex(mpreal(re, p), mpreal(im, p)) {}
    /// \brief Round or promote an existing complex value explicitly.
    mpcomplex(mpcomplex const& other, Precision p) : mpcomplex(p) { mpc_set(value_, other.native_handle(), MPC_RNDNN); }
    /// \brief Copy a valid borrowed MPC value at explicit component precision.
    /// \pre value points to an initialized MPC object during this call.
    explicit mpcomplex(mpc_srcptr value, Precision p) : mpcomplex(p) { mpc_set(value_, value, MPC_RNDNN); }
    mpcomplex(mpcomplex const& other)
    {
      if (other.initialized())
      {
        mpcomplex copy(other, other.precision());
        this->swap(copy);
      }
    }
    /// \brief Transfer ownership, leaving the source unset.
    mpcomplex(mpcomplex&& other) noexcept { this->swap(other); }
    mpcomplex& operator=(mpcomplex other) noexcept
    {
      this->swap(other);
      return *this;
    }
    ~mpcomplex()
    {
      if (initialized_) mpc_clear(value_);
    }

    bool initialized() const noexcept { return initialized_; }
    void swap(mpcomplex& other) noexcept
    {
      // Transfer the handle and its lifetime flag together, including unset state.
      std::swap(value_[0], other.value_[0]);
      std::swap(initialized_, other.initialized_);
    }
    Precision precision() const { return Precision::bits(mpc_get_prec(this->native_handle())); }
    /// \brief Borrow the initialized MPC value; unset access throws std::logic_error.
    mpc_srcptr native_handle() const
    {
      if (!initialized_) throw std::logic_error("complex<mpreal>: numerical use of an unset value");
      return value_;
    }
    mpcomplex at(Precision p) const { return mpcomplex(*this, p); }
    /// \brief Return an owning copy of the real component at this value's precision.
    mpreal real() const { return mpreal(mpc_realref(this->native_handle()), this->precision()); }
    /// \brief Return an owning copy of the imaginary component at this value's precision.
    mpreal imag() const { return mpreal(mpc_imagref(this->native_handle()), this->precision()); }
    /// \brief Set the real component, converting to the existing complex precision.
    void real(mpreal const& re)
    {
      (void)this->native_handle();
      mpfr_set(mpc_realref(value_), re.native_handle(), MPFR_RNDN);
    }
    /// \brief Set the imaginary component, converting to the existing complex precision.
    void imag(mpreal const& im)
    {
      (void)this->native_handle();
      mpfr_set(mpc_imagref(value_), im.native_handle(), MPFR_RNDN);
    }

    mpcomplex& operator+=(mpcomplex const& rhs);
    mpcomplex& operator-=(mpcomplex const& rhs);
    mpcomplex& operator*=(mpcomplex const& rhs);
    mpcomplex& operator/=(mpcomplex const& rhs);
    mpcomplex& operator+=(exact_constant const& rhs);
    mpcomplex& operator-=(exact_constant const& rhs);
    mpcomplex& operator*=(exact_constant const& rhs);
    mpcomplex& operator/=(exact_constant const& rhs);

  private:
    friend struct detail::mpcomplex_access;
    static Precision matching_precision(mpreal const& a, mpreal const& b)
    {
      if (a.precision() != b.precision())
        throw std::invalid_argument("complex<mpreal>: components require matching precision or explicit conversion");
      return a.precision();
    }
    mpc_t value_{};
    bool initialized_ = false;
};

namespace detail
{
struct mpcomplex_access
{
    template <auto Operation> static mpcomplex unary(mpcomplex const& x)
    {
      mpcomplex out(x.precision());
      Operation(out.value_, x.native_handle(), MPC_RNDNN);
      return out;
    }
    template <auto Operation> static mpcomplex& assign(mpcomplex& a, mpcomplex const& b)
    {
      if (a.precision() != b.precision())
        throw std::invalid_argument("complex<mpreal>: mixed precisions require explicit conversion");
      Operation(a.value_, a.value_, b.native_handle(), MPC_RNDNN);
      return a;
    }
    template <auto Operation> static mpcomplex binary(mpcomplex const& a, mpcomplex const& b)
    {
      (void)a.native_handle();
      mpcomplex out(a);
      assign<Operation>(out, b);
      return out;
    }
    template <auto Operation> static mpreal real_result(mpcomplex const& x)
    {
      mpreal out(x.precision());
      Operation(out.value_, x.native_handle(), MPFR_RNDN);
      return out;
    }
};
} // namespace detail

inline mpcomplex operator+(mpcomplex const& a, mpcomplex const& b)
{
  return detail::mpcomplex_access::binary<mpc_add>(a, b);
}
inline mpcomplex operator-(mpcomplex const& a, mpcomplex const& b)
{
  return detail::mpcomplex_access::binary<mpc_sub>(a, b);
}
inline mpcomplex operator*(mpcomplex const& a, mpcomplex const& b)
{
  return detail::mpcomplex_access::binary<mpc_mul>(a, b);
}
inline mpcomplex operator/(mpcomplex const& a, mpcomplex const& b)
{
  return detail::mpcomplex_access::binary<mpc_div>(a, b);
}
inline mpcomplex operator-(mpcomplex const& x) { return detail::mpcomplex_access::unary<mpc_neg>(x); }
inline mpcomplex operator+(mpcomplex const& x)
{
  (void)x.native_handle();
  return x;
}
inline mpcomplex& mpcomplex::operator+=(mpcomplex const& rhs)
{
  return detail::mpcomplex_access::assign<mpc_add>(*this, rhs);
}
inline mpcomplex& mpcomplex::operator-=(mpcomplex const& rhs)
{
  return detail::mpcomplex_access::assign<mpc_sub>(*this, rhs);
}
inline mpcomplex& mpcomplex::operator*=(mpcomplex const& rhs)
{
  return detail::mpcomplex_access::assign<mpc_mul>(*this, rhs);
}
inline mpcomplex& mpcomplex::operator/=(mpcomplex const& rhs)
{
  return detail::mpcomplex_access::assign<mpc_div>(*this, rhs);
}

inline mpcomplex operator+(mpcomplex const& a, exact_constant const& b) { return a + mpcomplex(b.at(a.precision())); }
inline mpcomplex operator-(mpcomplex const& a, exact_constant const& b) { return a - mpcomplex(b.at(a.precision())); }
inline mpcomplex operator*(mpcomplex const& a, exact_constant const& b) { return a * mpcomplex(b.at(a.precision())); }
inline mpcomplex operator/(mpcomplex const& a, exact_constant const& b) { return a / mpcomplex(b.at(a.precision())); }
inline mpcomplex operator+(exact_constant const& a, mpcomplex const& b) { return mpcomplex(a.at(b.precision())) + b; }
inline mpcomplex operator-(exact_constant const& a, mpcomplex const& b) { return mpcomplex(a.at(b.precision())) - b; }
inline mpcomplex operator*(exact_constant const& a, mpcomplex const& b) { return mpcomplex(a.at(b.precision())) * b; }
inline mpcomplex operator/(exact_constant const& a, mpcomplex const& b) { return mpcomplex(a.at(b.precision())) / b; }
inline mpcomplex& mpcomplex::operator+=(exact_constant const& rhs)
{
  return *this += mpcomplex(rhs.at(this->precision()));
}
inline mpcomplex& mpcomplex::operator-=(exact_constant const& rhs)
{
  return *this -= mpcomplex(rhs.at(this->precision()));
}
inline mpcomplex& mpcomplex::operator*=(exact_constant const& rhs)
{
  return *this *= mpcomplex(rhs.at(this->precision()));
}
inline mpcomplex& mpcomplex::operator/=(exact_constant const& rhs)
{
  return *this /= mpcomplex(rhs.at(this->precision()));
}

inline bool operator==(mpcomplex const& a, mpcomplex const& b)
{
  auto x = a.native_handle();
  auto y = b.native_handle();
  return mpfr_equal_p(mpc_realref(x), mpc_realref(y)) && mpfr_equal_p(mpc_imagref(x), mpc_imagref(y));
}
inline bool operator==(mpcomplex const& a, exact_constant const& b) { return a.real() == b && a.imag() == 0; }
inline mpcomplex conj(mpcomplex const& z) { return detail::mpcomplex_access::unary<mpc_conj>(z); }
inline mpreal abs(mpcomplex const& z) { return detail::mpcomplex_access::real_result<mpc_abs>(z); }
inline mpreal norm(mpcomplex const& z) { return detail::mpcomplex_access::real_result<mpc_norm>(z); }
inline mpreal arg(mpcomplex const& z) { return detail::mpcomplex_access::real_result<mpc_arg>(z); }
inline mpreal real(mpcomplex const& z) { return z.real(); }
inline mpreal imag(mpcomplex const& z) { return z.imag(); }
inline bool isfinite(mpcomplex const& z) { return isfinite(z.real()) && isfinite(z.imag()); }
inline bool isnan(mpcomplex const& z) { return isnan(z.real()) || isnan(z.imag()); }
inline bool isinf(mpcomplex const& z) { return isinf(z.real()) || isinf(z.imag()); }
/// \brief Principal square root, with the sign of imaginary zero selecting the cut side.
inline mpcomplex sqrt(mpcomplex const& z) { return detail::mpcomplex_access::unary<mpc_sqrt>(z); }
inline mpcomplex exp(mpcomplex const& z) { return detail::mpcomplex_access::unary<mpc_exp>(z); }
/// \brief Principal logarithm, with the sign of imaginary zero selecting the cut side.
inline mpcomplex log(mpcomplex const& z) { return detail::mpcomplex_access::unary<mpc_log>(z); }
inline mpcomplex sin(mpcomplex const& z) { return detail::mpcomplex_access::unary<mpc_sin>(z); }
inline mpcomplex cos(mpcomplex const& z) { return detail::mpcomplex_access::unary<mpc_cos>(z); }
inline mpcomplex pow(mpcomplex const& a, mpcomplex const& b) { return detail::mpcomplex_access::binary<mpc_pow>(a, b); }
inline std::ostream& operator<<(std::ostream& out, mpcomplex const& z)
{
  return out << '(' << z.real() << ',' << z.imag() << ')';
}
} // namespace uni20
