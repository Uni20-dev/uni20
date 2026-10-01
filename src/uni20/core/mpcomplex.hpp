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
struct exact_complex
{
    exact_constant re;
    exact_constant im;
};
struct mpc_value
{
    mpc_t value{};
    bool initialized = false;
    explicit mpc_value(Precision p)
    {
      require_mpfr_tls();
      mpc_init2(value, p.bit_count());
      initialized = true;
      mpc_set_ui(value, 0, MPC_RNDNN);
    }
    mpc_value(mpc_value const& other) : mpc_value(Precision::bits(mpc_get_prec(other.value)))
    {
      mpc_set(value, other.value, MPC_RNDNN);
    }
    mpc_value(mpc_value&& other) noexcept
    {
      std::swap(value[0], other.value[0]);
      std::swap(initialized, other.initialized);
    }
    mpc_value& operator=(mpc_value other) noexcept
    {
      std::swap(value[0], other.value[0]);
      std::swap(initialized, other.initialized);
      return *this;
    }
    ~mpc_value()
    {
      if (initialized) mpc_clear(value);
    }
};
} // namespace detail

/// \brief Exact rational complex value or owning finite-precision MPC approximation.
/// \details Both components are exact, or both have the same finite precision.
///          An exact component combined with an approximate component converts at
///          that component's precision. Arithmetic is eager; finite operations
///          round to nearest, ties to even. Default construction is exact zero.
class mpcomplex {
  public:
    using value_type = mpreal;
    /// \brief Construct exact complex zero.
    mpcomplex() : value_(std::in_place_type<detail::exact_complex>) {}
    /// \brief Construct an unset placeholder for overwrite storage.
    explicit mpcomplex(uninitialized_t) noexcept : value_(std::monostate{}) {}
    explicit mpcomplex(Precision p) : mpcomplex(uninitialized)
    {
      if (p.is_exact())
        value_.emplace<detail::exact_complex>();
      else
        value_.emplace<detail::mpc_value>(p);
    }
    /// \brief Construct components at explicit precision, or retain two exact components.
    mpcomplex(mpreal const& re, mpreal const& im, Precision p) : mpcomplex(p)
    {
      if (p.is_exact())
      {
        if (!re.precision().is_exact() || !im.precision().is_exact())
          throw std::invalid_argument("complex<mpreal>: an approximation cannot be reclassified as exact");
        value_ = detail::exact_complex{re.exact_value(), im.exact_value()};
      }
      else
      {
        re.copy_to(mpc_realref(this->approximate()));
        im.copy_to(mpc_imagref(this->approximate()));
      }
    }
    mpcomplex(mpreal const& re, mpreal const& im) : mpcomplex(re, im, common_precision(re.precision(), im.precision()))
    {}
    mpcomplex(mpreal const& re) : mpcomplex(re, mpreal{}, re.precision()) {}
    mpcomplex(mpreal const& re, Precision p) : mpcomplex(re, mpreal{}, p) {}
    template <std::integral I>
      requires(!std::same_as<I, bool>)
    explicit mpcomplex(I value) : mpcomplex(mpreal(value))
    {}
    explicit mpcomplex(exact_constant const& value) : mpcomplex(mpreal(value)) {}
    explicit mpcomplex(decimal_literal value) : mpcomplex(mpreal(value)) {}
    template <std::integral I>
      requires(!std::same_as<I, bool>)
    mpcomplex(I value, Precision p) : mpcomplex(mpreal(value, p))
    {}
    mpcomplex(exact_constant const& value, Precision p) : mpcomplex(mpreal(value, p)) {}
    mpcomplex(decimal_literal value, Precision p) : mpcomplex(mpreal(value, p)) {}
    mpcomplex(std::string_view re, std::string_view im, Precision p) : mpcomplex(mpreal(re, p), mpreal(im, p)) {}
    mpcomplex(mpcomplex const& other, Precision p) : mpcomplex(p)
    {
      other.require_value();
      if (p.is_exact())
      {
        if (!other.is_exact())
          throw std::invalid_argument("complex<mpreal>: an approximation cannot be reclassified as exact");
        value_ = std::get<detail::exact_complex>(other.value_);
      }
      else
        other.copy_to(this->approximate());
    }
    /// \brief Copy a valid borrowed MPC value at finite component precision.
    /// \pre value points to an initialized MPC object during this call.
    explicit mpcomplex(mpc_srcptr value, Precision p) : mpcomplex(p) { mpc_set(this->approximate(), value, MPC_RNDNN); }
    mpcomplex(mpcomplex const&) = default;
    /// \brief Transfer ownership, leaving the source unset.
    mpcomplex(mpcomplex&& other) noexcept : value_(std::move(other.value_)) { other.value_.emplace<std::monostate>(); }
    mpcomplex& operator=(mpcomplex other) noexcept
    {
      this->swap(other);
      return *this;
    }
    void swap(mpcomplex& other) noexcept { value_.swap(other.value_); }
    /// \brief Whether this object holds exact or approximate numerical components.
    bool initialized() const noexcept { return !std::holds_alternative<std::monostate>(value_); }
    /// \brief Whether both components are exact rationals.
    bool is_exact() const noexcept { return std::holds_alternative<detail::exact_complex>(value_); }
    Precision precision() const
    {
      this->require_value();
      return this->is_exact() ? Precision::exact() : Precision::bits(mpc_get_prec(this->native_handle()));
    }
    /// \brief Borrow an approximate MPC value; exact and unset states reject native access.
    mpc_srcptr native_handle() const
    {
      this->require_value();
      if (this->is_exact())
        throw std::logic_error("complex<mpreal>: exact value needs finite precision before MPC access");
      return std::get<detail::mpc_value>(value_).value;
    }
    /// \brief Round into an initialized MPC destination at its existing component precisions.
    void copy_to(mpc_ptr destination) const
    {
      this->require_value();
      if (this->is_exact())
      {
        auto const& q = std::get<detail::exact_complex>(value_);
        mpfr_set_q(mpc_realref(destination), q.re.native_handle(), MPFR_RNDN);
        mpfr_set_q(mpc_imagref(destination), q.im.native_handle(), MPFR_RNDN);
      }
      else
        mpc_set(destination, this->native_handle(), MPC_RNDNN);
    }
    mpcomplex at(Precision p) const { return mpcomplex(*this, p); }
    /// \brief Return an owning component with the same exact/approximate state.
    mpreal real() const
    {
      if (this->is_exact()) return mpreal(std::get<detail::exact_complex>(value_).re);
      return mpreal(mpc_realref(this->native_handle()), this->precision());
    }
    mpreal imag() const
    {
      if (this->is_exact()) return mpreal(std::get<detail::exact_complex>(value_).im);
      return mpreal(mpc_imagref(this->native_handle()), this->precision());
    }
    /// \brief Replace a component, preserving finite precision or adopting the first approximate component's precision.
    void real(mpreal const& re)
    {
      auto p = this->precision();
      if (p.is_exact())
        *this = mpcomplex(re, this->imag());
      else
        re.copy_to(mpc_realref(this->approximate()));
    }
    void imag(mpreal const& im)
    {
      auto p = this->precision();
      if (p.is_exact())
        *this = mpcomplex(this->real(), im);
      else
        im.copy_to(mpc_imagref(this->approximate()));
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
    explicit mpcomplex(detail::exact_complex q) : value_(std::move(q)) {}
    void require_value() const
    {
      if (!this->initialized()) throw std::logic_error("complex<mpreal>: numerical use of an unset value");
    }
    mpc_ptr approximate()
    {
      (void)this->native_handle();
      return std::get<detail::mpc_value>(value_).value;
    }
    std::variant<std::monostate, detail::exact_complex, detail::mpc_value> value_;
};

namespace detail
{
struct mpcomplex_access
{
    template <auto Operation> static mpcomplex unary(mpcomplex const& x, Precision p)
    {
      (void)p.bit_count();
      return unary<Operation>(x.at(p));
    }
    template <auto Operation> static mpcomplex binary(mpcomplex const& x, mpcomplex const& y, Precision p)
    {
      (void)p.bit_count();
      return binary<Operation>(x.at(p), y.at(p));
    }
    template <auto Operation> static mpreal real_result(mpcomplex const& x, Precision p)
    {
      (void)p.bit_count();
      return real_result<Operation>(x.at(p));
    }

    static mpcomplex exact_sqrt(mpcomplex const& x)
    {
      auto const& q = std::get<exact_complex>(x.value_);
      auto magnitude = uni20::sqrt(q.re * q.re + q.im * q.im);
      auto re = uni20::sqrt((magnitude + q.re) / 2);
      auto im = uni20::sqrt((magnitude - q.re) / 2);
      // Exact zero has no sign: the principal root uses the upper cut side.
      if (q.im < 0) im = -im;
      return mpcomplex(exact_complex{std::move(re), std::move(im)});
    }
    template <auto Operation> static mpcomplex unary(mpcomplex const& x)
    {
      auto p = x.precision();
      if (p.is_exact())
      {
        auto const& q = std::get<exact_complex>(x.value_);
        if constexpr (Operation == mpc_neg)
          return mpcomplex(exact_complex{-q.re, -q.im});
        else if constexpr (Operation == mpc_conj)
          return mpcomplex(exact_complex{q.re, -q.im});
        else if constexpr (Operation == mpc_sqrt)
          return exact_sqrt(x);
        else
        {
          if constexpr (Operation == mpc_exp || Operation == mpc_cos)
          {
            if (q.re == 0 && q.im == 0) return mpcomplex{1};
          }
          else if constexpr (Operation == mpc_log)
          {
            if (q.re == 1 && q.im == 0) return mpcomplex{};
          }
          else if constexpr (Operation == mpc_sin)
          {
            if (q.re == 0 && q.im == 0) return mpcomplex{};
          }
          throw std::logic_error("complex<mpreal>: elementary function requires finite working precision");
        }
      }
      mpcomplex out(p);
      Operation(out.approximate(), x.native_handle(), MPC_RNDNN);
      return out;
    }
    template <auto Operation> static mpcomplex binary(mpcomplex const& a, mpcomplex const& b)
    {
      auto p = common_precision(a.precision(), b.precision());
      if (p.is_exact())
      {
        auto const& x = std::get<exact_complex>(a.value_);
        auto const& y = std::get<exact_complex>(b.value_);
        if constexpr (Operation == mpc_add)
          return mpcomplex(exact_complex{x.re + y.re, x.im + y.im});
        else if constexpr (Operation == mpc_sub)
          return mpcomplex(exact_complex{x.re - y.re, x.im - y.im});
        else if constexpr (Operation == mpc_mul)
          return mpcomplex(exact_complex{x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re});
        else if constexpr (Operation == mpc_div)
        {
          auto d = y.re * y.re + y.im * y.im;
          return mpcomplex(exact_complex{(x.re * y.re + x.im * y.im) / d, (x.im * y.re - x.re * y.im) / d});
        }
        else
        {
          if constexpr (Operation == mpc_pow)
          {
            if (y.im == 0)
            {
              auto exponent = y.re.native_handle();
              if (mpz_cmp_ui(mpq_denref(exponent), 1) == 0) return integer_power(a, mpq_numref(exponent));
              if (mpz_cmp_ui(mpq_denref(exponent), 2) == 0) return integer_power(exact_sqrt(a), mpq_numref(exponent));
              if (x.im == 0 && x.re >= 0) return mpcomplex(uni20::pow(x.re, y.re));
            }
            if (x.re == 1 && x.im == 0) return mpcomplex{1};
          }
          throw std::logic_error("complex<mpreal>: elementary function requires finite working precision");
        }
      }
      mpcomplex av(uninitialized), bv(uninitialized);
      if (a.is_exact()) av = a.at(p);
      if (b.is_exact()) bv = b.at(p);
      mpcomplex out(p);
      Operation(out.approximate(), a.is_exact() ? av.native_handle() : a.native_handle(),
                b.is_exact() ? bv.native_handle() : b.native_handle(), MPC_RNDNN);
      return out;
    }
    template <auto Operation> static mpcomplex& assign(mpcomplex& a, mpcomplex const& b)
    {
      if (!a.is_exact() && !b.is_exact())
      {
        (void)common_precision(a.precision(), b.precision());
        Operation(a.approximate(), a.native_handle(), b.native_handle(), MPC_RNDNN);
      }
      else
        a = binary<Operation>(a, b);
      return a;
    }
    template <auto Operation> static mpreal real_result(mpcomplex const& x)
    {
      auto p = x.precision();
      if (p.is_exact())
      {
        if constexpr (Operation == mpc_norm)
        {
          auto const& q = std::get<exact_complex>(x.value_);
          return mpreal(q.re * q.re + q.im * q.im);
        }
        else
        {
          auto const& q = std::get<exact_complex>(x.value_);
          if constexpr (Operation == mpc_abs)
            return mpreal(uni20::sqrt(q.re * q.re + q.im * q.im));
          else if constexpr (Operation == mpc_arg)
            if (q.im == 0 && q.re > 0) return mpreal{};
          throw std::logic_error("complex<mpreal>: elementary function requires finite working precision");
        }
      }
      mpreal out(p);
      Operation(out.approximate(), x.native_handle(), MPFR_RNDN);
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
  (void)x.precision();
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

inline mpcomplex operator+(mpcomplex const& a, exact_constant const& b) { return a + mpcomplex(b); }
inline mpcomplex operator-(mpcomplex const& a, exact_constant const& b) { return a - mpcomplex(b); }
inline mpcomplex operator*(mpcomplex const& a, exact_constant const& b) { return a * mpcomplex(b); }
inline mpcomplex operator/(mpcomplex const& a, exact_constant const& b) { return a / mpcomplex(b); }
inline mpcomplex operator+(exact_constant const& a, mpcomplex const& b) { return mpcomplex(a) + b; }
inline mpcomplex operator-(exact_constant const& a, mpcomplex const& b) { return mpcomplex(a) - b; }
inline mpcomplex operator*(exact_constant const& a, mpcomplex const& b) { return mpcomplex(a) * b; }
inline mpcomplex operator/(exact_constant const& a, mpcomplex const& b) { return mpcomplex(a) / b; }
inline mpcomplex& mpcomplex::operator+=(exact_constant const& rhs) { return *this += mpcomplex(rhs); }
inline mpcomplex& mpcomplex::operator-=(exact_constant const& rhs) { return *this -= mpcomplex(rhs); }
inline mpcomplex& mpcomplex::operator*=(exact_constant const& rhs) { return *this *= mpcomplex(rhs); }
inline mpcomplex& mpcomplex::operator/=(exact_constant const& rhs) { return *this /= mpcomplex(rhs); }

inline bool operator==(mpcomplex const& a, mpcomplex const& b) { return a.real() == b.real() && a.imag() == b.imag(); }
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
/// \brief Evaluate `sqrt(z.at(p))`, returning an approximation at finite precision p.
/// \details MPC's principal branches and signed-zero rules apply after input conversion.
/// \throws std::logic_error If p is exact or z is unset.
inline mpcomplex sqrt(mpcomplex const& z, Precision p) { return detail::mpcomplex_access::unary<mpc_sqrt>(z, p); }
/// \brief Evaluate `exp(z.at(p))`, returning an approximation at finite precision p.
/// \details MPC's principal branches and signed-zero rules apply after input conversion.
/// \throws std::logic_error If p is exact or z is unset.
inline mpcomplex exp(mpcomplex const& z, Precision p) { return detail::mpcomplex_access::unary<mpc_exp>(z, p); }
/// \brief Evaluate `log(z.at(p))`, returning an approximation at finite precision p.
/// \details MPC's principal branches and signed-zero rules apply after input conversion.
/// \throws std::logic_error If p is exact or z is unset.
inline mpcomplex log(mpcomplex const& z, Precision p) { return detail::mpcomplex_access::unary<mpc_log>(z, p); }
/// \brief Evaluate `sin(z.at(p))`, returning an approximation at finite precision p.
/// \details MPC's principal branches and signed-zero rules apply after input conversion.
/// \throws std::logic_error If p is exact or z is unset.
inline mpcomplex sin(mpcomplex const& z, Precision p) { return detail::mpcomplex_access::unary<mpc_sin>(z, p); }
/// \brief Evaluate `cos(z.at(p))`, returning an approximation at finite precision p.
/// \details MPC's principal branches and signed-zero rules apply after input conversion.
/// \throws std::logic_error If p is exact or z is unset.
inline mpcomplex cos(mpcomplex const& z, Precision p) { return detail::mpcomplex_access::unary<mpc_cos>(z, p); }
/// \brief Evaluate `abs(z.at(p))`, returning an approximation at finite precision p.
/// \details MPC's principal branches and signed-zero rules apply after input conversion.
/// \throws std::logic_error If p is exact or z is unset.
inline mpreal abs(mpcomplex const& z, Precision p) { return detail::mpcomplex_access::real_result<mpc_abs>(z, p); }
/// \brief Evaluate `norm(z.at(p))`, returning an approximation at finite precision p.
/// \details MPC's principal branches and signed-zero rules apply after input conversion.
/// \throws std::logic_error If p is exact or z is unset.
inline mpreal norm(mpcomplex const& z, Precision p) { return detail::mpcomplex_access::real_result<mpc_norm>(z, p); }
/// \brief Evaluate `arg(z.at(p))`, returning an approximation at finite precision p.
/// \details MPC's principal branches and signed-zero rules apply after input conversion.
/// \throws std::logic_error If p is exact or z is unset.
inline mpreal arg(mpcomplex const& z, Precision p) { return detail::mpcomplex_access::real_result<mpc_arg>(z, p); }
/// \brief Evaluate the principal power `pow(a.at(p), b.at(p))` at finite precision p.
/// \throws std::logic_error If p is exact or either operand is unset.
inline mpcomplex pow(mpcomplex const& a, mpcomplex const& b, Precision p)
{
  return detail::mpcomplex_access::binary<mpc_pow>(a, b, p);
}

inline std::ostream& operator<<(std::ostream& out, mpcomplex const& z)
{
  return out << '(' << z.real() << ',' << z.imag() << ')';
}
} // namespace uni20
