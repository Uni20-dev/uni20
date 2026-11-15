// dense_hotrg.hpp -- a dense 2D HOTRG on the dense Tensor API, for ising_hotrg_example.cpp.
//
// HOTRG (Xie, Chen, Qin, Zhu, Yang and Xiang, Phys. Rev. B 86, 045139 (2012)) coarse-grains a
// square-lattice tensor network one axis at a time: two site tensors are joined along that axis
// and the pair of legs on each side of the other axis is squeezed back to one leg by an
// isometry.  The isometry here is the leading eigenspace of the gram of one such pair, taken
// through a truncated SVD, with a single global cut -- no symmetry, no per-sector allocation.
//
// A site tensor is ALWAYS stored as T[ym, xm, xp, yp], named by the face each leg sits on:
//
//     ym   the face on the low side of the y axis  (up)       written u below
//     xm   the face on the low side of the x axis  (left)     written l
//     xp   the face on the high side of the x axis (right)    written r
//     yp   the face on the high side of the y axis (down)     written d
//
// The kernel is the step ALONG y; the step along x is the kernel on the axis-exchanged tensor.
//
// Uni20 has no dense axis permutation and `contract` never packs an operand, so (1) every
// contraction below is written as a matrix product -- the surviving group leading or trailing,
// the summed group adjacent on both operands -- which is what reaches GEMM, and (2) every
// regrouping in between goes through a BlockTensor over single-sector U(1) spaces, whose
// `permute<>` + materialization is the permutation the dense Tensor lacks (`detail::wrap`,
// `detail::permuted`).  That API is capped at order 4: a rank-5 tensor enters it with two
// adjacent axes fused by `reshape_view`.
//
// Scalar-generic real arithmetic with explicit working precision, unit cell 1x1.
// `ring_levels` reads the transfer-matrix spectrum of the coarse tensor back out,
// which the example compares against exact Ising values
// (ising_strip_reference.hpp).
#pragma once

#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/reshape.hpp>
#include <uni20/tensor/tensor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20::examples::dense_hotrg
{

using idx = uni20::index_type;
using axis_pair = std::pair<std::size_t, std::size_t>;
template <uni20::Real R> using tensor3 = uni20::Tensor<R, 3>;
template <uni20::Real R> using tensor4 = uni20::Tensor<R, 4>;

enum class axis
{
  y,
  x
};
enum class model
{
  clock,
  potts
};

// Where a run's time goes, accumulated over every call since the program started: the GEMM
// contractions, the regroupings through the U(1) detour (strided copies in Uni20's reference
// loop), and the SVD of the gram.
struct seconds_spent
{
    double contract = 0.0, regroup = 0.0, svd = 0.0;
};

inline seconds_spent& spent()
{
  static seconds_spent totals;
  return totals;
}

namespace detail
{

struct lap
{
    double& slot;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    explicit lap(double& target) : slot(target) {}
    lap(lap const&) = delete;
    lap& operator=(lap const&) = delete;
    ~lap() { slot += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};

// Four legs, all in the domain, so that every permutation of them is side-preserving.
template <uni20::Real R>
using Wrap =
    uni20::BlockTensor<R, uni20::Domain<uni20::BlockSpace, uni20::BlockSpace, uni20::BlockSpace, uni20::BlockSpace>,
                       uni20::Codomain<>, uni20::PackedSparseBlockStorage<>>;

// U(1) with only its identity sector in use: a Wrap is one dense block.
struct trivial_u1
{
    uni20::Symmetry sym{"N:U(1)"};
    uni20::QNum q0 = uni20::QNum::identity(sym);

    [[nodiscard]] uni20::BlockSpace space(idx dim) const
    {
      return uni20::BlockSpace(sym, {{q0, static_cast<std::size_t>(dim)}}, "group");
    }
    [[nodiscard]] static auto key() { return uni20::BlockKey<4>{{0, 0, 0, 0}}; }
};

inline trivial_u1 const& group()
{
  static trivial_u1 const instance;
  return instance;
}

// An order-4 BlockTensor of the given extents. Each element is supplied by copy or
// a beta=0 contraction before it is read, including runtime-precision values.
template <uni20::Real R> [[nodiscard]] inline Wrap<R> wrap(idx e0, idx e1, idx e2, idx e3)
{
  auto const& g = group();
  return Wrap<R>(uni20::uninitialized, g.sym, uni20::Domain{g.space(e0), g.space(e1), g.space(e2), g.space(e3)},
                 uni20::Codomain<>{}, {trivial_u1::key()});
}

template <uni20::Real R> [[nodiscard]] inline Wrap<R> wrap(tensor4<R> const& t)
{
  lap const timer(spent().regroup);
  Wrap<R> w = wrap<R>(t.extent(0), t.extent(1), t.extent(2), t.extent(3));
  auto block = w.block(trivial_u1::key());
  uni20::copy(block, t);
  return w;
}

// The block with its axes regrouped, as an owning dense tensor: output axis j is source axis Axis[j].
template <std::size_t... Axis, uni20::Real R> [[nodiscard]] auto permuted(Wrap<R>& w, uni20::Precision p)
{
  lap const timer(spent().regroup);
  auto view = uni20::permute<Axis...>(w).block(trivial_u1::key());
  tensor4<R> result(uni20::uninitialized, view.extent(0), view.extent(1), view.extent(2), view.extent(3), p);
  uni20::copy(result, view);
  return result;
}

inline void require(bool ok, char const* what)
{
  if (!ok) throw std::invalid_argument(std::string("dense_hotrg: ") + what);
}

} // namespace detail

// ---------------------------------------------------------------------------------------------
// models
// ---------------------------------------------------------------------------------------------

/// \brief A tensor and the logarithm of the positive scalar multiplying it.
/// \details The physical vertex is exp(log_scale) * tensor; the exponential is never formed.
template <uni20::Real R> struct scaled_tensor
{
    tensor4<R> tensor;
    R log_scale;
};

namespace detail
{
template <uni20::Real R> R pi_value(uni20::Precision p)
{
  if constexpr (requires { uni20::pi<R>.at(p); })
    return uni20::pi<R>.at(p);
  else
    return uni20::pi<R>;
}
} // namespace detail

/// \brief Bond Fourier weights with exp(K) factored out, for ferromagnetic K > 0.
/// \details The 1/sqrt(q) factor makes the contracted vertex network the partition function.
///          Clock: sum_s cos(2*pi*m*s/q) exp(K*(cos(2*pi*s/q)-1)) / sqrt(q).
///          Potts: lambda_0 = (1+(q-1)exp(-K))/sqrt(q), lambda_m = (1-exp(-K))/sqrt(q).
template <uni20::Real R> [[nodiscard]] std::vector<R> bond_weights(model which, int q, R const& K)
{
  detail::require(q >= 2 && uni20::isfinite(K) && K > 0, "finite ferromagnetic K > 0 and q >= 2 required");
  auto const p = uni20::precision_of(K);
  detail::require(!p.is_exact(), "finite working precision required");
  R const one = uni20::scalar_at_precision<R>(1, p);
  R const root = uni20::math::sqrt(uni20::scalar_at_precision<R>(q, p));
  std::vector<R> lambda(static_cast<std::size_t>(q), uni20::scalar_at_precision<R>(0, p));
  for (int m = 0; m < q; ++m)
  {
    if (which == model::potts || q == 2 || q == 3)
    {
      R const exponent = which == model::potts ? K : (q == 2 ? 2 * K : K + K / 2);
      lambda[static_cast<std::size_t>(m)] =
          (m == 0 ? one + (q - 1) * uni20::math::exp(-exponent) : -uni20::math::expm1(-exponent)) / root;
    }
    else if (q == 4)
    {
      R const e = uni20::math::exp(-K), difference = -uni20::math::expm1(-K);
      lambda[static_cast<std::size_t>(m)] =
          (m == 0 ? (one + e) * (one + e) : (m == 2 ? difference * difference : -uni20::math::expm1(-2 * K))) / 2;
    }
    else
      for (int spin = 0; spin < q; ++spin)
      {
        R const angle = one * spin / q;
        R const sine = uni20::math::sinpi(angle);
        lambda[static_cast<std::size_t>(m)] +=
            uni20::math::cospi(2 * m * angle) * uni20::math::exp(-2 * K * sine * sine) / root;
      }
  }
  for (R const& value : lambda)
    detail::require(uni20::isfinite(value) && value >= 0, "bond Fourier weight unresolved at the working precision");
  return lambda;
}

//
//                 u
//                 | sqrt(lambda_u)
//            +----+----+
//  sqrt   l -|  delta  |- r   sqrt
//            +----+----+
//                 | sqrt(lambda_d)
//                 d
//
/// \brief Charge-basis clock/Potts vertex with the scale exp(2K) stored logarithmically.
/// \details T[u,l,r,d] = delta(u+l-r-d mod q) sqrt(lambda_u lambda_l lambda_r lambda_d).
///          Both the tensor elements and its log scale retain the precision of K.
template <uni20::Real R> [[nodiscard]] scaled_tensor<R> vertex(model which, int q, R const& K)
{
  auto const lambda = bond_weights(which, q, K);
  auto const p = uni20::precision_of(K);
  tensor4<R> t(q, q, q, q, p);
  for (idx up = 0; up < q; ++up)
    for (idx left = 0; left < q; ++left)
      for (idx right = 0; right < q; ++right)
        for (idx down = 0; down < q; ++down)
          if (((up + left - right - down) % q + q) % q == 0)
            t[up, left, right, down] = uni20::math::sqrt(lambda[up] * lambda[left] * lambda[right] * lambda[down]);
  R log_scale = 2 * K;
  detail::require(uni20::isfinite(log_scale),
                  "the physical log scale is not representable at this scalar's exponent range");
  return {std::move(t), std::move(log_scale)};
}

/// \brief Closed-form critical coupling J/Tc evaluated at the selected precision.
template <uni20::Real R> [[nodiscard]] R critical_coupling(model which, int q, uni20::Precision p)
{
  R const one = uni20::scalar_at_precision<R>(1, p);
  if (which == model::potts) return uni20::math::log1p(uni20::math::sqrt(one * q));
  if (q == 2) return uni20::math::log1p(uni20::math::sqrt(2 * one)) / 2;
  if (q == 3) return 2 * uni20::math::log1p(uni20::math::sqrt(3 * one)) / 3;
  if (q == 4) return uni20::math::log1p(uni20::math::sqrt(2 * one));
  throw std::invalid_argument("dense_hotrg: no closed-form Tc for this clock model");
}

// ---------------------------------------------------------------------------------------------
// small pieces
// ---------------------------------------------------------------------------------------------

// Coarse-step normalization: the largest positive entry. The initial normalization
// in flow uses the largest modulus to bound every entry before the first Gram matrix.
template <uni20::Real R> [[nodiscard]] inline R max_element(tensor4<R> const& t)
{
  auto const p = uni20::precision_of(t);
  R largest = uni20::scalar_at_precision<R>(0, p);
  for (auto const& value : t.storage())
    largest = std::max(largest, R(value));
  return largest;
}

template <uni20::Real R> inline void scale(tensor4<R>& t, R factor)
{
  for (auto& value : t.storage())
    value = R(value) * factor;
}

// sum_{i, j} T[i, j, j, i]: the tensor closed on itself in both directions (one block on a torus).
template <uni20::Real R> [[nodiscard]] inline R torus_trace(tensor4<R> const& t)
{
  auto const p = uni20::precision_of(t);
  detail::require(t.extent(0) == t.extent(3) && t.extent(1) == t.extent(2),
                  "torus_trace: opposite faces differ in dimension");
  R sum = uni20::scalar_at_precision<R>(0, p);
  for (idx i = 0; i < t.extent(0); ++i)
    for (idx j = 0; j < t.extent(1); ++j)
      sum += R(t[i, j, j, i]);
  return sum;
}

// The same tensor with the two axes' names exchanged, T[u, l, r, d] -> [l, u, d, r].
template <uni20::Real R> [[nodiscard]] inline tensor4<R> exchange_axes(tensor4<R> const& t)
{
  auto const p = uni20::precision_of(t);
  auto w = detail::wrap(t);
  return detail::permuted<1, 0, 3, 2>(w, p);
}

// ---------------------------------------------------------------------------------------------
// the kernel: the step along y
// ---------------------------------------------------------------------------------------------

// The gram of the fused xm pair, theta[l1, l2, l1', l2']: the two-site column against itself,
// every leg but the two left ones contracted.
//
//            ┌────────────── u ─────────────┐
//            │                              │
//     ┌──────┴──────┐                ┌──────┴──────┐
//  l1 ┤      T      ├ r ──────── r ──┤      T      ├ l1'       A[l1, s, l1', s'] = sum_{u, r}  T T
//     └──────┬──────┘                └──────┬──────┘
//            │ s                            │ s'
//     ┌──────┴──────┐                ┌──────┴──────┐
//  l2 ┤      T      ├ r ──────── r ──┤      T      ├ l2'       B[l2, s, l2', s'] = sum_{r, d}  T T
//     └──────┬──────┘                └──────┬──────┘
//            │                              │
//            └────────────── d ─────────────┘
//
//   theta[(l1, l2) ; (l1', l2')] = sum_{s, s'} A B         (three chi^6 products)
template <uni20::Real R> [[nodiscard]] inline tensor4<R> gram_y(tensor4<R> const& t)
{
  auto const p = uni20::precision_of(t);
  idx const m = t.extent(0), n = t.extent(1);
  detail::require(t.extent(2) == n && t.extent(3) == m, "gram: opposite faces differ in dimension");
  auto const key = detail::trivial_u1::key();
  using std::array;

  auto site = detail::wrap(t);
  auto const t_ld_ur = detail::permuted<1, 3, 0, 2>(site, p); // [l, d, u, r]
  auto const t_lu_rd = detail::permuted<1, 0, 2, 3>(site, p); // [l, u, r, d]

  auto upper = detail::wrap<R>(n, m, n, m); // A[l1, s, l1', s'], s the d face of the upper copy
  {
    auto block = upper.block(key);
    auto out = uni20::reshape_view(block, n, m, n, m);
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(out, uni20::scalar_at_precision<R>(1, p), t_ld_ur, t_ld_ur,
                            array<axis_pair, 2>{{{2, 2}, {3, 3}}}, uni20::scalar_at_precision<R>(0, p), p);
  }
  auto lower = detail::wrap<R>(n, m, n, m); // B[l2, s, l2', s'], s the u face of the lower copy
  {
    auto block = lower.block(key);
    auto out = uni20::reshape_view(block, n, m, n, m);
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(out, uni20::scalar_at_precision<R>(1, p), t_lu_rd, t_lu_rd,
                            array<axis_pair, 2>{{{2, 2}, {3, 3}}}, uni20::scalar_at_precision<R>(0, p), p);
  }
  auto const a = detail::permuted<0, 2, 1, 3>(upper, p); // [l1, l1', s, s']
  auto const b = detail::permuted<1, 3, 0, 2>(lower, p); // [s, s', l2, l2']

  auto paired = detail::wrap<R>(n, n, n, n); // [l1, l1', l2, l2']
  {
    auto block = paired.block(key);
    auto out = uni20::reshape_view(block, n, n, n, n);
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(out, uni20::scalar_at_precision<R>(1, p), a, b, array<axis_pair, 2>{{{2, 0}, {3, 1}}},
                            uni20::scalar_at_precision<R>(0, p), p);
  }
  return detail::permuted<0, 2, 1, 3>(paired, p); // [l1, l2, l1', l2']
}

// The update network with a given isometry U[a, b, k] (a from the upper copy, b from the lower):
//
//               u                                    u
//               │                                    │
//     ┌───┐  ┌──┴───┐  ┌───┐              ┌──────────┴──────────┐
//  l ─┤   ├──┤  Tu  ├──┤   ├─ r           │                     │
//     │ U │  └──┬───┘  │ U │       =   l ─┤         T'          ├─ r
//     │   │     │ s    │   │              │                     │
//     │   ├──┐  │   ┌──┤   │              └──────────┬──────────┘
//     └───┘  │┌─┴───┐│ └───┘                         │
//            └┤  Td  ├┘                              d
//             └──┬───┘
//                d
//
//   T'[u, k1, k2, d] = sum U[l1, l2, k1] Tu[u, l1, r1, s] Td[s, l2, r2, d] U[r1, r2, k2]
//
// in the order ((Tu U) Td) U: chi^6, chi^7, chi^6 in time and chi^5 in memory.  U is real, so
// the U^dagger of the left pair is U itself.
template <uni20::Real R>
[[nodiscard]] inline tensor4<R> apply_isometry(tensor4<R> const& upper, tensor4<R> const& lower, tensor3<R> const& iso)
{
  auto const p = uni20::precision_of(upper);
  idx const mu = upper.extent(0), n = upper.extent(1), ms = upper.extent(3), md = lower.extent(3), k = iso.extent(2);
  detail::require(upper.extent(2) == n && lower.extent(1) == n && lower.extent(2) == n && lower.extent(0) == ms,
                  "apply_isometry: the two tensors do not stack");
  detail::require(iso.extent(0) == n && iso.extent(1) == n, "apply_isometry: the isometry does not fit the fused pair");
  auto const key = detail::trivial_u1::key();
  using std::array;

  auto up = detail::wrap(upper);
  auto const tu = detail::permuted<0, 2, 3, 1>(up, p); // [u, r1, s, l1]
  auto low = detail::wrap(lower);
  auto const td = detail::permuted<1, 0, 2, 3>(low, p); // [l2, s, r2, d]

  // H2[u, r1, k1, r2, d] = sum_{l2, s} H1 Td, held as [(u r1), k1, r2, d].  The scopes keep at
  // most two chi^5 buffers alive at a time.
  auto second = [&] {
    // H1[u, r1, s, l2, k1] = sum_{l1} Tu U, held as [(u r1), s, l2, k1]
    auto const h1_owner = [&] {
      auto first = detail::wrap<R>(mu * n, ms, n, k);
      {
        auto block = first.block(key);
        auto out = uni20::reshape_view(block, mu, n, ms, n, k);
        detail::lap const timer(spent().contract);
        uni20::linalg::contract(out, uni20::scalar_at_precision<R>(1, p), tu, iso, array<axis_pair, 1>{{{3, 0}}},
                                uni20::scalar_at_precision<R>(0, p), p);
      }
      return detail::permuted<0, 3, 2, 1>(first, p); // [(u r1), k1, l2, s]
    }();
    auto const h1 = uni20::reshape_view(h1_owner, mu, n, k, n, ms);
    auto held = detail::wrap<R>(mu * n, k, n, md);
    {
      auto block = held.block(key);
      auto out = uni20::reshape_view(block, mu, n, k, n, md);
      detail::lap const timer(spent().contract);
      uni20::linalg::contract(out, uni20::scalar_at_precision<R>(1, p), h1, td, array<axis_pair, 2>{{{3, 0}, {4, 1}}},
                              uni20::scalar_at_precision<R>(0, p), p);
    }
    return held;
  }();
  auto const h2_owner = detail::permuted<1, 3, 0, 2>(second, p); // [k1, d, (u r1), r2]
  auto const h2 = uni20::reshape_view(h2_owner, k, md, mu, n, n);

  // [k1, d, u, k2] = sum_{r1, r2} H2 U
  auto third = detail::wrap<R>(k, md, mu, k);
  {
    auto block = third.block(key);
    auto out = uni20::reshape_view(block, k, md, mu, k);
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(out, uni20::scalar_at_precision<R>(1, p), h2, iso, array<axis_pair, 2>{{{3, 0}, {4, 1}}},
                            uni20::scalar_at_precision<R>(0, p), p);
  }
  return detail::permuted<2, 0, 3, 1>(third, p); // [u, k1, k2, d]
}

template <uni20::Real R> struct step_result
{
    tensor4<R> T;          // the coarse tensor [ym, xm, xp, yp], divided by `norm`
    tensor3<R> U;          // the isometry [a, b, k]: a the upper (left) copy's leg, b the lower (right) one's
    std::vector<R> S = {}; // the kept singular values of the gram, descending
    R norm = R{0};         // the largest entry the coarse tensor was divided by
    R discarded = R{0};    // 1 - sum(S) / tr(gram): the discarded fraction of the gram's trace
    R gap = R{0};          // (smallest kept - largest discarded) / smallest kept singular value; 1 if nothing was
                           // discarded, 0 if the smallest kept value is zero.  Near 0 the cut falls inside a
                           // degenerate cluster and the kept subspace is the solver's choice, not the model's.
};

namespace detail
{

template <uni20::Real R> [[nodiscard]] inline step_result<R> coarse_grain_y(tensor4<R> const& t, std::size_t D)
{
  auto const p = uni20::precision_of(t);
  idx const n = t.extent(1);
  tensor4<R> theta = gram_y(t);
  R trace = uni20::scalar_at_precision<R>(0, p);
  for (idx i = 0; i < n; ++i)
    for (idx j = 0; j < n; ++j)
      trace += R(theta[i, j, i, j]);

  std::size_t const rows = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  auto decomposition = [&] {
    lap const timer(spent().svd);
    return uni20::linalg::truncated_svd(
        uni20::reshape_view(theta, n * n, n * n),
        uni20::linalg::SvdTruncationPolicy<R>{.maximum_retained_extent = std::min(D, rows)}, p);
  }();
  idx const k = static_cast<idx>(decomposition.singular_values.extent(0));

  step_result<R> result{.T = tensor4<R>(1, 1, 1, 1, p), .U = tensor3<R>(n, n, k, p)};
  uni20::copy(result.U, uni20::reshape_view(decomposition.left_singular_vectors, n, n, k));
  R kept = uni20::scalar_at_precision<R>(0, p);
  for (idx i = 0; i < k; ++i)
  {
    result.S.push_back(R(decomposition.singular_values[i]));
    kept += result.S.back();
  }
  result.discarded = 1 - kept / trace;
  auto const& cut = decomposition.truncation;
  if (!cut.smallest_retained_singular_value || !cut.largest_discarded_singular_value)
    result.gap = uni20::scalar_at_precision<R>(1, p); // nothing was discarded
  else if (*cut.smallest_retained_singular_value > 0)
    result.gap = (*cut.smallest_retained_singular_value - *cut.largest_discarded_singular_value) /
                 *cut.smallest_retained_singular_value;
  else
    result.gap = uni20::scalar_at_precision<R>(
        0, p); // the cut falls inside the null space: kept and discarded values are both zero

  result.T = apply_isometry(t, t, result.U);
  result.norm = max_element(result.T);
  require(result.norm > 0, "coarse_grain: the largest entry of the coarse tensor is not positive");
  scale(result.T, R(1 / result.norm));
  return result;
}

} // namespace detail

// The gram of the fused low-face pair of the OTHER axis (xm for a y step, ym for an x step).
template <uni20::Real R> [[nodiscard]] inline tensor4<R> gram(tensor4<R> const& t, axis along)
{
  return along == axis::y ? gram_y(t) : gram_y(exchange_axes(t));
}

// One HOTRG step along `along` with a global top-D cut: gram -> truncated SVD -> the update
// network with U on both fused faces -> divide by the largest entry.  For the x step exchange
// the names ym <-> xm, yp <-> xp in every picture.
template <uni20::Real R>
[[nodiscard]] inline step_result<R> coarse_grain(tensor4<R> const& t, axis along, std::size_t D)
{
  if (along == axis::y) return detail::coarse_grain_y(t, D);
  step_result<R> result = detail::coarse_grain_y(exchange_axes(t), D);
  result.T = exchange_axes(result.T);
  return result;
}

// ---------------------------------------------------------------------------------------------
// the ladder
// ---------------------------------------------------------------------------------------------

/// \brief One record per round; the represented square has width 2^log2_width.
template <uni20::Real R> struct flow_record
{
    std::vector<int> log2_width = {};
    std::vector<R> log_norm_y = {}, log_norm_x = {};
    std::vector<R> discarded_y = {}, discarded_x = {}, gap_y = {}, gap_x = {};
    std::vector<R> lnz = {}; // log Z per original lattice site
    std::vector<double> seconds = {}, seconds_contract = {}, seconds_regroup = {}, seconds_svd = {};
    R initial_log_scale;
    R unit_lnz;
    tensor4<R> T0;
};

/// \brief Coarse-grain normalized tensors, retaining removed scales as logarithms.
/// \details One round doubles each side. No lattice extent or normalization product is formed.
///          The finite input tensor is normalized before its first Gram matrix; its
///          periodic traces must be positive for the partition-function logarithm.
///          The observer receives the current record and its normalized coarse tensor.
template <uni20::Real R, class Observer = std::nullptr_t>
[[nodiscard]] flow_record<R> flow(scaled_tensor<R> unit, std::size_t D, int steps, Observer observe = nullptr)
{
  detail::require(steps >= 0 && D > 0, "nonnegative rounds and positive bond dimension required");
  auto const p = uni20::precision_of(unit.tensor);
  detail::require(!p.is_exact(), "finite working precision required");
  R norm = uni20::scalar_at_precision<R>(0, p);
  for (auto const& value : unit.tensor.storage())
  {
    detail::require(uni20::isfinite(value), "the input tensor must be finite");
    norm = std::max(norm, uni20::math::abs(value));
  }
  detail::require(norm > 0 && uni20::isfinite(unit.log_scale), "nonzero tensor and finite log scale required");
  // Divide directly: the reciprocal of a tiny finite norm could overflow.
  for (auto& value : unit.tensor.storage())
    value /= norm;
  R const initial = unit.log_scale + uni20::math::log(norm);
  R const unit_lnz = initial + uni20::math::log(torus_trace(unit.tensor));
  flow_record<R> record{.initial_log_scale = initial, .unit_lnz = unit_lnz, .T0 = std::move(unit.tensor)};
  R sum = initial, weight = uni20::scalar_at_precision<R>(1, p);
  for (int step = 0; step < steps; ++step)
  {
    auto const start = std::chrono::steady_clock::now();
    seconds_spent const before = spent();
    auto along_y = coarse_grain(record.T0, axis::y, D);
    auto along_x = coarse_grain(along_y.T, axis::x, D);
    record.T0 = std::move(along_x.T);
    record.log2_width.push_back(step + 1);
    record.log_norm_y.push_back(uni20::math::log(along_y.norm));
    record.log_norm_x.push_back(uni20::math::log(along_x.norm));
    weight /= 4;
    sum += weight * (2 * record.log_norm_y.back() + record.log_norm_x.back());
    record.lnz.push_back(sum + weight * uni20::math::log(torus_trace(record.T0)));
    record.discarded_y.push_back(along_y.discarded);
    record.discarded_x.push_back(along_x.discarded);
    record.gap_y.push_back(along_y.gap);
    record.gap_x.push_back(along_x.gap);
    record.seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    record.seconds_contract.push_back(spent().contract - before.contract);
    record.seconds_regroup.push_back(spent().regroup - before.regroup);
    record.seconds_svd.push_back(spent().svd - before.svd);
    if constexpr (!std::same_as<Observer, std::nullptr_t>) observe(record, record.T0);
  }
  return record;
}

// One coarse tensor with its two horizontal legs joined:
//
//              ym
//              |
//          +---+---+
//       +--|   T   |--+
//       |  +---+---+  |
//       |      |      |
//       +--- xp = xm -+
//              yp
//
//      M[ym, yp] = sum over x of T[ym, x, x, yp]
//
// A tensor that spans `rows` lattice rows and `columns` lattice columns makes this the transfer
// matrix of a periodic strip of `columns` sites, advanced by `rows` rows at once: M = V^rows.
// After `k` HOTRG rounds from a vertex tensor, rows = columns = 2^k, so the eigenvalue ratios of
// M are the strip's own ratios raised to the width -- which is what `scaling_dimensions` undoes.
// Ratios are free of every normalization the flow divided out, so reading them needs no
// bookkeeping.
//
// This is a trace of one tensor against itself over a single index.  `contract` does not express
// it -- it joins two operands -- so it is written out; it costs one pass and no extra storage.
template <uni20::Real R> [[nodiscard]] inline uni20::Tensor<R, 2> ring_matrix(tensor4<R> const& t)
{
  auto const p = uni20::precision_of(t);
  idx const mu = t.extent(0), nh = t.extent(1), md = t.extent(3);
  detail::require(t.extent(2) == nh, "ring_matrix: the two horizontal legs differ in dimension");
  uni20::Tensor<R, 2> ring(mu, md, p); // zero-initialized
  for (idx bond = 0; bond < nh; ++bond)
    for (idx down = 0; down < md; ++down)
      for (idx up = 0; up < mu; ++up)
        ring[up, down] += R(t[up, bond, bond, down]);
  return ring;
}

template <uni20::Real R> struct ring_result
{
    std::vector<R> eigenvalues; ///< descending
    R asymmetry = R{0};         ///< max |M - M^T| / max |M|; zero when the tensor is reflection-symmetric
};

// The spectrum of `ring_matrix`.  The transfer matrix of a reflection-symmetric tensor is a
// symmetric matrix, which `asymmetry` reports and `eigh` then relies on: read it before the
// eigenvalues.
template <uni20::Real R> [[nodiscard]] inline ring_result<R> ring_levels(tensor4<R> const& t)
{
  auto const p = uni20::precision_of(t);
  uni20::Tensor<R, 2> ring = ring_matrix(t);
  idx const dim = ring.extent(0);
  detail::require(dim == ring.extent(1), "ring_levels: the ym and yp legs differ in dimension");

  ring_result<R> result;
  R largest = uni20::scalar_at_precision<R>(0, p), worst = largest;
  for (idx i = 0; i < dim; ++i)
    for (idx j = 0; j < dim; ++j)
    {
      largest = std::max(largest, uni20::math::abs(R(ring[i, j])));
      worst = std::max(worst, uni20::math::abs(R(ring[i, j]) - R(ring[j, i])));
    }
  result.asymmetry = largest > 0 ? worst / largest : uni20::scalar_at_precision<R>(0, p);

  auto const decomposition = uni20::linalg::eigh(ring, p); // ascending
  result.eigenvalues.reserve(static_cast<std::size_t>(dim));
  for (idx i = dim; i > 0; --i)
    result.eigenvalues.push_back(R(decomposition.eigenvalues[i - 1]));
  return result;
}

/// \brief Scaling dimensions from a transfer matrix spanning powers of two in each direction.
/// \details The geometric factor uses only the difference of the width and height exponents.
///          Positive eigenvalues above 64 working-precision epsilons of the leading value are retained.
template <uni20::Real R>
[[nodiscard]] std::vector<R> scaling_dimensions(std::vector<R> const& eigenvalues, int log2_columns = 0,
                                                int log2_rows = 0)
{
  detail::require(!eigenvalues.empty() && eigenvalues.front() > 0, "no positive leading eigenvalue");
  detail::require(log2_columns >= 0 && log2_rows >= 0, "negative lattice extent exponent");
  auto const p = uni20::precision_of(eigenvalues.front());
  R const one = uni20::scalar_at_precision<R>(1, p);
  R const floor = 64 * uni20::numeric_limits<R>::epsilon(p);
  R const factor = uni20::math::scalbn(one, log2_columns - log2_rows) / (2 * detail::pi_value<R>(p));
  std::vector<R> out;
  for (R const& value : eigenvalues)
  {
    if (!(value > floor * eigenvalues.front())) break;
    R const gap = value > eigenvalues.front() / 2
                      ? -uni20::math::log1p((value - eigenvalues.front()) / eigenvalues.front())
                      : uni20::math::log(eigenvalues.front()) - uni20::math::log(value);
    out.push_back(gap * factor);
  }
  return out;
}

/// \brief Onsager log Z per site, using bounded exponentials and midpoint quadrature.
/// \details Arithmetic retains K's precision; quadrature discretization is independently controlled by points.
template <uni20::Real R> [[nodiscard]] R onsager_lnz(R const& K, int points = 400000)
{
  detail::require(uni20::isfinite(K) && K > 0 && points > 0, "positive finite K and quadrature count required");
  auto const p = uni20::precision_of(K);
  R const one = uni20::scalar_at_precision<R>(1, p);
  R const e = uni20::math::exp(-2 * K), e2 = e * e;
  R const kappa = std::min(one, R(4 * e * (one - e2) / ((one + e2) * (one + e2))));
  R const complement = (one - kappa) * (one + kappa);
  R const denominator = 2 * (one * points);
  R const spacing = detail::pi_value<R>(p) / denominator;
  R sum = uni20::scalar_at_precision<R>(0, p);
  for (int i = 0; i < points; ++i)
  {
    R const theta = (one * i + one / 2) * spacing;
    R const cosine = uni20::math::cos(theta);
    R const root = uni20::math::sqrt(complement + kappa * kappa * cosine * cosine);
    sum += uni20::math::log1p((root - one) / 2);
  }
  return 2 * K + uni20::math::log1p(e2) + sum / denominator;
}

} // namespace uni20::examples::dense_hotrg
