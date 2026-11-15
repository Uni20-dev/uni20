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
// Real double precision, unit cell 1x1.  `ring_levels` reads the transfer-matrix spectrum of
// the coarse tensor back out, which is what the example compares against exact Ising values
// (ising_strip_reference.hpp).
#pragma once

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
using tensor3 = uni20::Tensor<double, 3>;
using tensor4 = uni20::Tensor<double, 4>;

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
using Wrap =
    uni20::BlockTensor<double,
                       uni20::Domain<uni20::BlockSpace, uni20::BlockSpace, uni20::BlockSpace, uni20::BlockSpace>,
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
    [[nodiscard]] static Wrap::key_type key() { return Wrap::key_type{{0, 0, 0, 0}}; }
};

inline trivial_u1 const& group()
{
  static trivial_u1 const instance;
  return instance;
}

// An order-4 BlockTensor of the given extents; its one block is where `contract` writes.
[[nodiscard]] inline Wrap wrap(idx e0, idx e1, idx e2, idx e3)
{
  auto const& g = group();
  return Wrap(g.sym, uni20::Domain{g.space(e0), g.space(e1), g.space(e2), g.space(e3)}, uni20::Codomain<>{},
              {trivial_u1::key()});
}

[[nodiscard]] inline Wrap wrap(tensor4 const& t)
{
  lap const timer(spent().regroup);
  Wrap w = wrap(t.extent(0), t.extent(1), t.extent(2), t.extent(3));
  auto block = w.block(trivial_u1::key());
  uni20::copy(block, t);
  return w;
}

// The block with its axes regrouped, as an owning dense tensor: output axis j is source axis Axis[j].
template <std::size_t... Axis> [[nodiscard]] auto permuted(Wrap& w)
{
  lap const timer(spent().regroup);
  return uni20::make_tensor<uni20::ColumnMajor>(uni20::permute<Axis...>(w).block(trivial_u1::key()));
}

inline void require(bool ok, char const* what)
{
  if (!ok) throw std::invalid_argument(std::string("dense_hotrg: ") + what);
}

} // namespace detail

// ---------------------------------------------------------------------------------------------
// models
// ---------------------------------------------------------------------------------------------

// The bond weights lambda_m, m = 0 .. q-1, of a Z_q model at coupling K = J / T.  The 1/sqrt(q)
// is carried here rather than on the vertex, so that the network of vertices is the partition
// function with no further factor:
//    clock  lambda_m = sum_s cos(2 pi m s / q) exp(K cos(2 pi s / q)) / sqrt(q),
//    Potts  lambda_0 = (e^K + q - 1) / sqrt(q),  lambda_m = (e^K - 1) / sqrt(q).
[[nodiscard]] inline std::vector<double> bond_weights(model which, int q, double K)
{
  detail::require(q >= 2, "q must be at least 2");
  std::vector<double> lambda(static_cast<std::size_t>(q), 0.0);
  double const root = std::sqrt(static_cast<double>(q));
  for (int m = 0; m < q; ++m)
  {
    if (which == model::potts)
      lambda[static_cast<std::size_t>(m)] = (m == 0 ? std::exp(K) + q - 1 : std::exp(K) - 1.0) / root;
    else
      for (int s = 0; s < q; ++s)
      {
        double const theta = 2.0 * std::numbers::pi * s / q;
        lambda[static_cast<std::size_t>(m)] += std::cos(m * theta) * std::exp(K * std::cos(theta)) / root;
      }
  }
  for (double value : lambda)
    detail::require(value > 0.0, "a bond weight is not positive (ferromagnetic K > 0 only)");
  return lambda;
}

// The charge-basis vertex tensor:
//
//                 u
//                 │ sqrt(lambda_u)
//            ┌────┴────┐
//  sqrt   l ─┤  delta  ├─ r   sqrt        T[u, l, r, d] = delta(u + l - r - d = 0 mod q)
//            └────┬────┘                                  sqrt(lambda_u lambda_l lambda_r lambda_d)
//                 │ sqrt(lambda_d)
//                 d
//
// With the 1/sqrt(q) in lambda the network of these tensors IS the partition function.
[[nodiscard]] inline tensor4 vertex(model which, int q, double K)
{
  auto const lambda = bond_weights(which, q, K);
  tensor4 t(q, q, q, q);
  for (idx up = 0; up < q; ++up)
    for (idx left = 0; left < q; ++left)
      for (idx right = 0; right < q; ++right)
        for (idx down = 0; down < q; ++down)
        {
          bool const allowed = ((up + left - right - down) % q + q) % q == 0;
          t[up, left, right, down] =
              allowed ? std::sqrt(lambda[static_cast<std::size_t>(up)] * lambda[static_cast<std::size_t>(left)] *
                                  lambda[static_cast<std::size_t>(right)] * lambda[static_cast<std::size_t>(down)])
                      : 0.0;
        }
  return t;
}

// The critical temperature T_c = J / K_c where it is known in closed form.
[[nodiscard]] inline double critical_temperature(model which, int q)
{
  if (which == model::potts) return 1.0 / std::log(1.0 + std::sqrt(static_cast<double>(q)));
  if (q == 2) return 2.0 / std::log(1.0 + std::sqrt(2.0));
  if (q == 3) return 3.0 / (2.0 * std::log(1.0 + std::sqrt(3.0)));
  if (q == 4) return 1.0 / std::log(1.0 + std::sqrt(2.0));
  throw std::invalid_argument("dense_hotrg: no closed-form T_c for the clock model at this q");
}

// ---------------------------------------------------------------------------------------------
// small pieces
// ---------------------------------------------------------------------------------------------

// The normalization used throughout: the largest ENTRY (not the largest modulus, not a norm).
// Only ratios are read from the flow, so the choice has to be consistent, not canonical.
[[nodiscard]] inline double max_element(tensor4 const& t)
{
  double largest = -HUGE_VAL;
  for (auto const& value : t.storage())
    largest = std::max(largest, static_cast<double>(value));
  return largest;
}

inline void scale(tensor4& t, double factor)
{
  for (auto& value : t.storage())
    value = static_cast<double>(value) * factor;
}

// sum_{i, j} T[i, j, j, i]: the tensor closed on itself in both directions (one block on a torus).
[[nodiscard]] inline double torus_trace(tensor4 const& t)
{
  detail::require(t.extent(0) == t.extent(3) && t.extent(1) == t.extent(2),
                  "torus_trace: opposite faces differ in dimension");
  double sum = 0.0;
  for (idx i = 0; i < t.extent(0); ++i)
    for (idx j = 0; j < t.extent(1); ++j)
      sum += static_cast<double>(t[i, j, j, i]);
  return sum;
}

// The same tensor with the two axes' names exchanged, T[u, l, r, d] -> [l, u, d, r].
[[nodiscard]] inline tensor4 exchange_axes(tensor4 const& t)
{
  auto w = detail::wrap(t);
  return detail::permuted<1, 0, 3, 2>(w);
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
[[nodiscard]] inline tensor4 gram_y(tensor4 const& t)
{
  idx const m = t.extent(0), n = t.extent(1);
  detail::require(t.extent(2) == n && t.extent(3) == m, "gram: opposite faces differ in dimension");
  auto const key = detail::trivial_u1::key();
  using std::array;

  auto site = detail::wrap(t);
  auto const t_ld_ur = detail::permuted<1, 3, 0, 2>(site); // [l, d, u, r]
  auto const t_lu_rd = detail::permuted<1, 0, 2, 3>(site); // [l, u, r, d]

  auto upper = detail::wrap(n, m, n, m); // A[l1, s, l1', s'], s the d face of the upper copy
  {
    auto block = upper.block(key);
    auto out = uni20::reshape_view(block, n, m, n, m);
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(out, 1.0, t_ld_ur, t_ld_ur, array<axis_pair, 2>{{{2, 2}, {3, 3}}}, 0.0);
  }
  auto lower = detail::wrap(n, m, n, m); // B[l2, s, l2', s'], s the u face of the lower copy
  {
    auto block = lower.block(key);
    auto out = uni20::reshape_view(block, n, m, n, m);
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(out, 1.0, t_lu_rd, t_lu_rd, array<axis_pair, 2>{{{2, 2}, {3, 3}}}, 0.0);
  }
  auto const a = detail::permuted<0, 2, 1, 3>(upper); // [l1, l1', s, s']
  auto const b = detail::permuted<1, 3, 0, 2>(lower); // [s, s', l2, l2']

  auto paired = detail::wrap(n, n, n, n); // [l1, l1', l2, l2']
  {
    auto block = paired.block(key);
    auto out = uni20::reshape_view(block, n, n, n, n);
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(out, 1.0, a, b, array<axis_pair, 2>{{{2, 0}, {3, 1}}}, 0.0);
  }
  return detail::permuted<0, 2, 1, 3>(paired); // [l1, l2, l1', l2']
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
[[nodiscard]] inline tensor4 apply_isometry(tensor4 const& upper, tensor4 const& lower, tensor3 const& iso)
{
  idx const mu = upper.extent(0), n = upper.extent(1), ms = upper.extent(3), md = lower.extent(3), k = iso.extent(2);
  detail::require(upper.extent(2) == n && lower.extent(1) == n && lower.extent(2) == n && lower.extent(0) == ms,
                  "apply_isometry: the two tensors do not stack");
  detail::require(iso.extent(0) == n && iso.extent(1) == n, "apply_isometry: the isometry does not fit the fused pair");
  auto const key = detail::trivial_u1::key();
  using std::array;

  auto up = detail::wrap(upper);
  auto const tu = detail::permuted<0, 2, 3, 1>(up); // [u, r1, s, l1]
  auto low = detail::wrap(lower);
  auto const td = detail::permuted<1, 0, 2, 3>(low); // [l2, s, r2, d]

  // H2[u, r1, k1, r2, d] = sum_{l2, s} H1 Td, held as [(u r1), k1, r2, d].  The scopes keep at
  // most two chi^5 buffers alive at a time.
  auto second = [&] {
    // H1[u, r1, s, l2, k1] = sum_{l1} Tu U, held as [(u r1), s, l2, k1]
    auto const h1_owner = [&] {
      auto first = detail::wrap(mu * n, ms, n, k);
      {
        auto block = first.block(key);
        auto out = uni20::reshape_view(block, mu, n, ms, n, k);
        detail::lap const timer(spent().contract);
        uni20::linalg::contract(out, 1.0, tu, iso, array<axis_pair, 1>{{{3, 0}}}, 0.0);
      }
      return detail::permuted<0, 3, 2, 1>(first); // [(u r1), k1, l2, s]
    }();
    auto const h1 = uni20::reshape_view(h1_owner, mu, n, k, n, ms);
    auto held = detail::wrap(mu * n, k, n, md);
    {
      auto block = held.block(key);
      auto out = uni20::reshape_view(block, mu, n, k, n, md);
      detail::lap const timer(spent().contract);
      uni20::linalg::contract(out, 1.0, h1, td, array<axis_pair, 2>{{{3, 0}, {4, 1}}}, 0.0);
    }
    return held;
  }();
  auto const h2_owner = detail::permuted<1, 3, 0, 2>(second); // [k1, d, (u r1), r2]
  auto const h2 = uni20::reshape_view(h2_owner, k, md, mu, n, n);

  // [k1, d, u, k2] = sum_{r1, r2} H2 U
  auto third = detail::wrap(k, md, mu, k);
  {
    auto block = third.block(key);
    auto out = uni20::reshape_view(block, k, md, mu, k);
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(out, 1.0, h2, iso, array<axis_pair, 2>{{{3, 0}, {4, 1}}}, 0.0);
  }
  return detail::permuted<2, 0, 3, 1>(third); // [u, k1, k2, d]
}

struct step_result
{
    tensor4 T;                  // the coarse tensor [ym, xm, xp, yp], divided by `norm`
    tensor3 U;                  // the isometry [a, b, k]: a the upper (left) copy's leg, b the lower (right) one's
    std::vector<double> S = {}; // the kept singular values of the gram, descending
    double norm = 0.0;          // the largest entry the coarse tensor was divided by
    double discarded = 0.0;     // 1 - sum(S) / tr(gram): the discarded fraction of the gram's trace
    double gap = 0.0;           // (smallest kept - largest discarded) / smallest kept singular value; 1 if nothing was
                                // discarded.  Near 0 the cut falls inside a degenerate cluster and the kept
                                // subspace is the solver's choice, not the model's.
};

namespace detail
{

[[nodiscard]] inline step_result coarse_grain_y(tensor4 const& t, std::size_t D)
{
  idx const n = t.extent(1);
  tensor4 theta = gram_y(t);
  double trace = 0.0;
  for (idx i = 0; i < n; ++i)
    for (idx j = 0; j < n; ++j)
      trace += static_cast<double>(theta[i, j, i, j]);

  std::size_t const rows = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  auto decomposition = [&] {
    lap const timer(spent().svd);
    return uni20::linalg::truncated_svd(
        uni20::reshape_view(theta, n * n, n * n),
        uni20::linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = std::min(D, rows)});
  }();
  idx const k = static_cast<idx>(decomposition.singular_values.extent(0));

  step_result result{.T = tensor4(1, 1, 1, 1), .U = tensor3(n, n, k)};
  uni20::copy(result.U, uni20::reshape_view(decomposition.left_singular_vectors, n, n, k));
  double kept = 0.0;
  for (idx i = 0; i < k; ++i)
  {
    result.S.push_back(static_cast<double>(decomposition.singular_values[i]));
    kept += result.S.back();
  }
  result.discarded = 1.0 - kept / trace;
  auto const& cut = decomposition.truncation;
  result.gap = cut.smallest_retained_singular_value && cut.largest_discarded_singular_value
                   ? (*cut.smallest_retained_singular_value - *cut.largest_discarded_singular_value) /
                         *cut.smallest_retained_singular_value
                   : 1.0;

  result.T = apply_isometry(t, t, result.U);
  result.norm = max_element(result.T);
  require(result.norm > 0.0, "coarse_grain: the largest entry of the coarse tensor is not positive");
  scale(result.T, 1.0 / result.norm);
  return result;
}

} // namespace detail

// The gram of the fused low-face pair of the OTHER axis (xm for a y step, ym for an x step).
[[nodiscard]] inline tensor4 gram(tensor4 const& t, axis along)
{
  return along == axis::y ? gram_y(t) : gram_y(exchange_axes(t));
}

// One HOTRG step along `along` with a global top-D cut: gram -> truncated SVD -> the update
// network with U on both fused faces -> divide by the largest entry.  For the x step exchange
// the names ym <-> xm, yp <-> xp in every picture.
[[nodiscard]] inline step_result coarse_grain(tensor4 const& t, axis along, std::size_t D)
{
  if (along == axis::y) return detail::coarse_grain_y(t, D);
  step_result result = detail::coarse_grain_y(exchange_axes(t), D);
  result.T = exchange_axes(result.T);
  return result;
}

// ---------------------------------------------------------------------------------------------
// the ladder
// ---------------------------------------------------------------------------------------------

// Everything a run yields, one entry per step L -> 2L (entry i builds L = 2^(i+1)).
struct flow_record
{
    std::vector<long long> L;
    std::vector<double> Ny, Nx, Norms; // the per-step norms; Norms = Ny^2 Nx
    std::vector<double> discarded_y, discarded_x, gap_y, gap_x;
    std::vector<double> trace_T;                                        // torus_trace of T0 at that L
    std::vector<double> lnz;                                            // ln Z / L^2 of the L x L torus
    std::vector<double> seconds;                                        // wall time of the step ...
    std::vector<double> seconds_contract, seconds_regroup, seconds_svd; // ... and where it went
    double trace_T_unit = 0.0;
    tensor4 T0 = tensor4(1, 1, 1, 1); // the last coarse tensor
};

// ln Z / L^2 after `count` steps: ln tr T_L / 4^count + sum_i ln(Norms_i) / 4^(i+1).  Each
// round quadruples the sites, so a norm divided out at step i is worth 4^-(i+1) per site.
[[nodiscard]] inline double lnz_per_site(std::vector<double> const& norms, std::size_t count, double trace)
{
  double sum = 0.0, weight = 1.0;
  for (std::size_t i = 0; i < count; ++i)
  {
    weight *= 0.25;
    sum += weight * std::log(norms[i]);
  }
  return sum + weight * std::log(trace);
}

// Called after every step with the record so far and the coarse tensor of that size.
using flow_observer = std::function<void(flow_record const&, tensor4 const& T0)>;

// The ladder, from the unit tensor up to L = 2^steps:
//
//     L -> 2L:    T1, U_y, N_y = coarse_grain(T0, y)        the y step
//                 T0, U_x, N_x = coarse_grain(T1, x)        the x step
//                 Norms.append(N_y^2 N_x)
[[nodiscard]] inline flow_record flow(tensor4 const& unit, std::size_t D, int steps, flow_observer const& observe = {})
{
  flow_record record;
  record.trace_T_unit = torus_trace(unit);
  record.T0 = unit;
  long long size = 1;
  for (int step = 0; step < steps; ++step)
  {
    auto const start = std::chrono::steady_clock::now();
    seconds_spent const before = spent();
    step_result along_y = coarse_grain(record.T0, axis::y, D);
    step_result along_x = coarse_grain(along_y.T, axis::x, D);
    record.T0 = std::move(along_x.T);
    size *= 2;

    record.L.push_back(size);
    record.Ny.push_back(along_y.norm);
    record.Nx.push_back(along_x.norm);
    record.Norms.push_back(along_y.norm * along_y.norm * along_x.norm);
    record.discarded_y.push_back(along_y.discarded);
    record.discarded_x.push_back(along_x.discarded);
    record.gap_y.push_back(along_y.gap);
    record.gap_x.push_back(along_x.gap);
    record.trace_T.push_back(torus_trace(record.T0));
    record.lnz.push_back(lnz_per_site(record.Norms, record.Norms.size(), record.trace_T.back()));
    record.seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    record.seconds_contract.push_back(spent().contract - before.contract);
    record.seconds_regroup.push_back(spent().regroup - before.regroup);
    record.seconds_svd.push_back(spent().svd - before.svd);
    if (observe) observe(record, record.T0);
  }
  return record;
}

// ---------------------------------------------------------------------------------------------
// the transfer matrix of the coarse tensor
// ---------------------------------------------------------------------------------------------

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
[[nodiscard]] inline uni20::Tensor<double, 2> ring_matrix(tensor4 const& t)
{
  idx const mu = t.extent(0), nh = t.extent(1), md = t.extent(3);
  detail::require(t.extent(2) == nh, "ring_matrix: the two horizontal legs differ in dimension");
  uni20::Tensor<double, 2> ring(mu, md); // zero-initialized
  for (idx bond = 0; bond < nh; ++bond)
    for (idx down = 0; down < md; ++down)
      for (idx up = 0; up < mu; ++up)
        ring[up, down] += static_cast<double>(t[up, bond, bond, down]);
  return ring;
}

struct ring_result
{
    std::vector<double> eigenvalues; ///< descending
    double asymmetry = 0.0;          ///< max |M - M^T| / max |M|; zero when the tensor is reflection-symmetric
};

// The spectrum of `ring_matrix`.  The transfer matrix of a reflection-symmetric tensor is a
// symmetric matrix, which `asymmetry` reports and `eigh` then relies on: read it before the
// eigenvalues.
[[nodiscard]] inline ring_result ring_levels(tensor4 const& t)
{
  uni20::Tensor<double, 2> ring = ring_matrix(t);
  idx const dim = ring.extent(0);
  detail::require(dim == ring.extent(1), "ring_levels: the ym and yp legs differ in dimension");

  ring_result result;
  double largest = 0.0, worst = 0.0;
  for (idx i = 0; i < dim; ++i)
    for (idx j = 0; j < dim; ++j)
    {
      largest = std::max(largest, std::abs(static_cast<double>(ring[i, j])));
      worst = std::max(worst, std::abs(static_cast<double>(ring[i, j]) - static_cast<double>(ring[j, i])));
    }
  result.asymmetry = largest > 0.0 ? worst / largest : 0.0;

  auto const decomposition = uni20::linalg::eigh(ring); // ascending
  result.eigenvalues.reserve(static_cast<std::size_t>(dim));
  for (idx i = dim - 1; i >= 0; --i)
    result.eigenvalues.push_back(static_cast<double>(decomposition.eigenvalues[i]));
  return result;
}

// X_i = (n / 2 pi rows) ln(lambda_0 / lambda_i) for a transfer matrix that spans `n` sites and
// `rows` rows: the scaling dimensions of the strip, as the ratios of `ring_levels` give them.
// For the ring of one coarse tensor n == rows and the factor is simply 1 / 2 pi.  Levels at or
// below `floor` times the largest are dropped, as are non-positive ones.
[[nodiscard]] inline std::vector<double> scaling_dimensions(std::vector<double> const& eigenvalues, long long n,
                                                            long long rows, double floor = 1e-13)
{
  detail::require(!eigenvalues.empty() && eigenvalues.front() > 0.0,
                  "scaling_dimensions: no positive leading eigenvalue");
  std::vector<double> out;
  for (double value : eigenvalues)
  {
    if (!(value > floor * eigenvalues.front())) break;
    out.push_back(std::log(eigenvalues.front() / value) * static_cast<double>(n) /
                  (2.0 * std::numbers::pi * static_cast<double>(rows)));
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// exact references
// ---------------------------------------------------------------------------------------------

// Onsager's ln Z / N of the square-lattice Ising model at coupling K (midpoint rule).
[[nodiscard]] inline double onsager_lnz(double K)
{
  double const kappa = 2.0 * std::sinh(2.0 * K) / (std::cosh(2.0 * K) * std::cosh(2.0 * K));
  int const points = 400000;
  double sum = 0.0;
  for (int i = 0; i < points; ++i)
  {
    double const theta = (i + 0.5) * (std::numbers::pi / 2.0) / points;
    double const sine = kappa * std::sin(theta);
    sum += std::log((1.0 + std::sqrt(1.0 - sine * sine)) / 2.0);
  }
  return std::log(2.0 * std::cosh(2.0 * K)) + 0.5 * sum / points;
}

} // namespace uni20::examples::dense_hotrg
