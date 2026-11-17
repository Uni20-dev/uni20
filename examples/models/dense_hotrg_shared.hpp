// dense_hotrg_shared.hpp -- the HOTRG step of dense_hotrg.hpp, written once, for host tensors and
// for CUDA tensors.
//
// The step is the algorithm of dense_hotrg.hpp: the gram of a fused leg pair from three chi^6
// products, its leading singular vectors, the update network ((Tu U) Td) U at chi^7, and a
// division by the largest entry.  Here every function is a template over the place where its
// tensors live, and the body is the same text for both places.  The device place exists in a
// build with the CUDA, cuBLAS and cuSOLVER backends; without them only the host place is
// compiled.
//
// What cannot be the same text is collected in the namespace `shim`, as pairs of overloads:
// the host overload is the idiom of dense_hotrg.hpp, and the device overload stands in for a
// function Uni20 does not offer for a CUDA tensor.  Each device overload says what is missing,
// what the host writes, and what is done instead.  Reading `shim` is reading the list of
// functions this step misses on the device:
//
//   M1  linalg::truncated_svd of a device matrix            -> cut_gram
//   M2  reshape_view of a device tensor                     -> cut_gram
//   M3  a reduction for the largest entry of a device tensor -> largest_entry
//   M4  a trace over paired axes of a device tensor         -> torus_trace (and the gram's trace in cut_gram)
//   M5  access to the entries of a device tensor            -> place, on_host (by design, not a defect)
//
// Two things dense_hotrg.hpp does are written differently here so that one text serves both
// places; neither needs a workaround:
//
//   * its 8 reshape_view calls around `contract` are gone.  They fused axes so that a chi^5
//     intermediate fitted a four-leg BlockTensor; a BlockTensor may have five legs (#132), so
//     the intermediates are held at their own order.
//   * scaling is `transform_inplace(t, linalg::scale{f})`, which exists for both places, instead of
//     a loop over the storage.
#pragma once

#include "dense_hotrg.hpp"

#include <uni20/config.hpp>

#if UNI20_BACKEND_CUDA && UNI20_BACKEND_CUBLAS && UNI20_BACKEND_CUSOLVER
#define UNI20_EXAMPLE_DENSE_HOTRG_DEVICE 1
#include <uni20/backend/cublas/execution.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/backend/cusolver/execution.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/storage/cuda_storage.hpp>
#else
#define UNI20_EXAMPLE_DENSE_HOTRG_DEVICE 0
#endif
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/tensor/copy_into.hpp>
#include <uni20/tensor/transform.hpp>

#include <type_traits>

namespace uni20::examples::dense_hotrg_shared
{

// dense_hotrg.hpp: its model tensors, its flow record and its readers (free energy, ring levels)
// are used as they are, and its step is the reference the checks compare against.
namespace reference = uni20::examples::dense_hotrg;
using reference::axis;
using reference::axis_pair;
using reference::idx;

inline uni20::Symmetry const& symmetry()
{
  static uni20::Symmetry const instance{"N:U(1)"};
  return instance;
}

// ---------------------------------------------------------------------------------------------
// the two places
// ---------------------------------------------------------------------------------------------

struct host_place
{
    template <std::size_t Rank> using tensor = uni20::Tensor<double, Rank>;
    using block_storage = uni20::PackedSparseBlockStorage<>;
    static constexpr char const* name = "host";

    template <class Wrap, class DomainType>
    [[nodiscard]] static Wrap construct(DomainType domain, typename Wrap::key_type key)
    {
      return Wrap(symmetry(), std::move(domain), uni20::Codomain<>{}, {key});
    }
    static void drain() {}
};

#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
struct device_place
{
    template <std::size_t Rank> using tensor = uni20::CudaTensor<double, Rank>;
    using block_storage = uni20::PackedSparseBlockStorage<uni20::CudaStorage>;
    static constexpr char const* name = "device";

    // The only difference from the host: a device tensor is told which device it lives on.
    template <class Wrap, class DomainType>
    [[nodiscard]] static Wrap construct(DomainType domain, typename Wrap::key_type key)
    {
      return Wrap(symmetry(), std::move(domain), uni20::Codomain<>{}, {key}, uni20::cuda::device_resources());
    }
    // Device work is queued; a timer has to wait for the queue to see the work it measures.
    static void drain() { uni20::cuda::device_resources().streams().synchronize(); }
};

template <class Tensor>
using place_of =
    std::conditional_t<std::same_as<uni20::tensor_storage_policy_t<std::remove_cvref_t<Tensor>>, uni20::CudaStorage>,
                       device_place, host_place>;
#else
template <class Tensor> using place_of = host_place;
#endif

// Where the device's gram is decomposed (see shim::cut_gram).  The host place ignores it.
enum class device_svd
{
  cusolver, // the full SVD on the device, cut on the host
  host      // the gram is downloaded and cut by the host truncated_svd
};

// Wall time by kind of work, accumulated since the program started.  `fallback` is the time the
// device spends in the M3 and M4 stand-ins, which go through the host.
struct seconds_spent
{
    double contract = 0.0, regroup = 0.0, svd = 0.0, fallback = 0.0;
};

inline seconds_spent& spent()
{
  static seconds_spent totals;
  return totals;
}

namespace detail
{

template <class Place> struct lap
{
    double& slot;
    std::chrono::steady_clock::time_point start;
    explicit lap(double& target) : slot(target)
    {
      Place::drain();
      start = std::chrono::steady_clock::now();
    }
    lap(lap const&) = delete;
    lap& operator=(lap const&) = delete;
    ~lap()
    {
      Place::drain();
      slot += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

// A BlockTensor whose legs carry only the U(1) identity sector is one dense block, and
// BlockTensor has the permutation the dense Tensor lacks (#130).  This is how dense_hotrg.hpp
// regroups; it works unchanged on device storage.
template <std::size_t> using leg_type = uni20::BlockSpace;
template <std::size_t... I> auto domain_of(std::index_sequence<I...>) -> uni20::Domain<leg_type<I>...>;
template <std::size_t Order> using domain_t = decltype(domain_of(std::make_index_sequence<Order>{}));
template <class Place, std::size_t Order>
using wrap_t = uni20::BlockTensor<double, domain_t<Order>, uni20::Codomain<>, typename Place::block_storage>;

#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
template <class Wrap>
using place_of_wrap = std::conditional_t<std::same_as<typename Wrap::storage_policy, device_place::block_storage>,
                                         device_place, host_place>;
#else
template <class Wrap> using place_of_wrap = host_place;
#endif

[[nodiscard]] inline uni20::BlockSpace leg(idx dim)
{
  return uni20::BlockSpace(symmetry(), {{uni20::QNum::identity(symmetry()), static_cast<std::size_t>(dim)}}, "group");
}

template <class Wrap> [[nodiscard]] typename Wrap::key_type key() { return typename Wrap::key_type{}; }

// A zero tensor of the given extents, as a single-sector BlockTensor.
template <class Place, class... Extents> [[nodiscard]] auto make_wrap(Extents... extents)
{
  using Wrap = wrap_t<Place, sizeof...(Extents)>;
  return Place::template construct<Wrap>(uni20::Domain{leg(static_cast<idx>(extents))...}, key<Wrap>());
}

// A dense tensor copied into a single-sector BlockTensor.
template <class Tensor> [[nodiscard]] auto wrap(Tensor const& t)
{
  using Place = place_of<Tensor>;
  constexpr std::size_t rank = uni20::tensor_mdspec_t<Tensor>::rank();
  lap<Place> const timer(spent().regroup);
  auto w = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return make_wrap<Place>(t.extent(I)...);
  }(std::make_index_sequence<rank>{});
  auto block = w.block(key<std::remove_cvref_t<decltype(w)>>());
  uni20::copy(block, t);
  return w;
}

// The block with its axes regrouped, as an owning dense tensor: output axis j is source axis Axis[j].
template <std::size_t... Axis> [[nodiscard]] auto permuted(auto& w)
{
  using Wrap = std::remove_cvref_t<decltype(w)>;
  lap<place_of_wrap<Wrap>> const timer(spent().regroup);
  return uni20::make_tensor<uni20::ColumnMajor>(uni20::permute<Axis...>(w).block(key<Wrap>()));
}

// out = sum over the paired axes of a and b, written into the block of `out`.  Every call of the
// step pairs axis groups that are adjacent in both operands, so it is one GEMM: OpenBLAS on the
// host, cuBLAS on the device.  (A contraction that is not one GEMM would run in cpu_reference on
// the host and has no backend on the device, where it ends the process.  The step never asks for one.)
template <class Wrap, class A, class B, std::size_t Pairs>
void contract_into(Wrap& out, A const& a, B const& b, std::array<axis_pair, Pairs> const& pairs)
{
  lap<place_of_wrap<Wrap>> const timer(spent().contract);
  auto block = out.block(key<Wrap>());
  uni20::linalg::contract(block, 1.0, a, b, pairs, 0.0);
}

} // namespace detail

// ---------------------------------------------------------------------------------------------
// shim: the places where the host and the device cannot be the same text
// ---------------------------------------------------------------------------------------------
namespace shim
{

using host4 = host_place::tensor<4>;
using host3 = host_place::tensor<3>;
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
using device4 = device_place::tensor<4>;
using device3 = device_place::tensor<3>;
#endif

// --- M5: access to the entries -----------------------------------------------------------------
// A device tensor has no t[i, j, k, l] and no iteration over its storage: its memory is not the
// host's.  This is by design.  A tensor is therefore built on the host and transferred, and read
// on the host after a transfer.  Both transfers are of chi^4 entries.

[[nodiscard]] inline host4 place(host_place, host4 const& t)
{
  host4 copy(t.extent(0), t.extent(1), t.extent(2), t.extent(3));
  uni20::copy(copy, t);
  return copy;
}
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
[[nodiscard]] inline device4 place(device_place, host4 const& t)
{
  return uni20::to_device(t, uni20::cuda::device_resources());
}
#endif

[[nodiscard]] inline host4 const& on_host(host_place, host4 const& t) { return t; }
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
[[nodiscard]] inline host4 on_host(device_place, device4 const& t) { return uni20::to_host(t); }
#endif

// --- M3: the largest entry ----------------------------------------------------------------------
// HOST IDIOM:  a loop over t.storage().
// MISSING:     Uni20 has no reduction for the largest entry (or the largest modulus) of a tensor.
//              The host can loop over the storage; the device cannot (M5).  norm_host(t) and
//              inner_product_host(a, b) exist for a device tensor, sum_host(t) does not compile.
// WORKAROUND:  download the tensor (chi^4 entries) and loop on the host.
// WOULD FIT:   a reduction returning a host scalar, as norm_host does; cuBLAS has idamax for the modulus.

[[nodiscard]] inline double largest_entry(host_place, host4 const& t) { return reference::max_element(t); }
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
[[nodiscard]] inline double largest_entry(device_place, device4 const& t)
{
  detail::lap<device_place> const timer(spent().fallback);
  return reference::max_element(uni20::to_host(t));
}
#endif

// --- M4: a trace over paired axes ---------------------------------------------------------------
// HOST IDIOM:  sum over i, j of t[i, j, j, i], by element access.
// MISSING:     no call takes a trace over paired axes of one tensor, on the host or on the device;
//              the host writes the loop, the device cannot (M5).
// WORKAROUND:  download the tensor and take the trace on the host.

[[nodiscard]] inline double torus_trace(host_place, host4 const& t) { return reference::torus_trace(t); }
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
[[nodiscard]] inline double torus_trace(device_place, device4 const& t)
{
  detail::lap<device_place> const timer(spent().fallback);
  return reference::torus_trace(uni20::to_host(t));
}
#endif

// --- M1, M2: the cut of the gram ----------------------------------------------------------------
template <class Place> struct cut
{
    typename Place::template tensor<3> U; // [a, b, k]: the leading singular vectors of the gram
    std::vector<double> S;                // the kept singular values, descending
    double discarded;                     // 1 - sum(S) / tr(gram)
    double gap;                           // (smallest kept - largest discarded) / smallest kept
};

[[nodiscard]] inline double relative_gap(double smallest_kept, bool anything_discarded, double largest_discarded)
{
  if (!anything_discarded) return 1.0;
  if (smallest_kept > 0.0) return (smallest_kept - largest_discarded) / smallest_kept;
  return 0.0;
}

// The host: the idiom of dense_hotrg.hpp.
[[nodiscard]] inline cut<host_place> cut_gram(host_place, host4 theta, idx n, std::size_t D, device_svd)
{
  detail::lap<host_place> const timer(spent().svd);
  double trace = 0.0;
  for (idx i = 0; i < n; ++i)
    for (idx j = 0; j < n; ++j)
      trace += static_cast<double>(theta[i, j, i, j]);
  std::size_t const rows = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  auto decomposition = uni20::linalg::truncated_svd(
      uni20::reshape_view(theta, n * n, n * n),
      uni20::linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = std::min(D, rows)});
  idx const k = static_cast<idx>(decomposition.singular_values.extent(0));
  host3 U(n, n, k);
  uni20::copy(U, uni20::reshape_view(decomposition.left_singular_vectors, n, n, k));
  std::vector<double> S;
  double kept = 0.0;
  for (idx i = 0; i < k; ++i)
  {
    S.push_back(static_cast<double>(decomposition.singular_values[i]));
    kept += S.back();
  }
  auto const& truncation = decomposition.truncation;
  bool const discarded_any = truncation.smallest_retained_singular_value && truncation.largest_discarded_singular_value;
  double const gap = relative_gap(discarded_any ? *truncation.smallest_retained_singular_value : 0.0, discarded_any,
                                  discarded_any ? *truncation.largest_discarded_singular_value : 0.0);
  return {.U = std::move(U), .S = std::move(S), .discarded = 1.0 - kept / trace, .gap = gap};
}

#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
// The device.
//
// HOST IDIOM:  truncated_svd(reshape_view(theta, n * n, n * n), policy), then
//              copy(U, reshape_view(left_singular_vectors, n, n, k)), and the trace of theta by
//              element access.
// MISSING M1:  linalg::truncated_svd has no implementation for a device matrix (a static assertion
//              says so and names to_host()).  linalg::eigh and linalg::self_adjoint_eigh, which
//              would suit a symmetric gram, have none either.
// MISSING M2:  reshape_view does not accept a device tensor, so the gram cannot be viewed as a
//              matrix, nor the vectors as a three-leg tensor.
// WORKAROUND:  * the gram is an owning tensor here, so reshape(std::move(theta), n * n, n * n)
//                turns it into a matrix without a copy;
//              * linalg::singular_value_decomposition, which does run on the device (cuSOLVER
//                gesvd), gives all n^2 singular values and both factors;
//              * the values and the left factor are downloaded, the leading k columns are copied
//                into U on the host, and U (chi^3 entries) is uploaded;
//              * the gram is positive semidefinite, so its trace is the sum of all its singular
//                values, which replaces the element access (M4).
// COST:        the right factor and n^2 - k singular vectors are computed and thrown away.  This
//              SVD is about half of the device step at bond 64.
// ALTERNATIVE: device_svd::host downloads the gram and uses the host cut above, which is what the
//              static assertion suggests.  It is far slower at a large bond, because the host SVD
//              (LAPACK gesvd) is.
[[nodiscard]] inline cut<device_place> cut_gram(device_place, device4 theta, idx n, std::size_t D, device_svd route)
{
  auto& resources = uni20::cuda::device_resources();
  if (route == device_svd::host)
  {
    cut<host_place> on_host = cut_gram(host_place{}, uni20::to_host(theta), n, D, route);
    return {.U = uni20::to_device(on_host.U, resources),
            .S = std::move(on_host.S),
            .discarded = on_host.discarded,
            .gap = on_host.gap};
  }
  detail::lap<device_place> const timer(spent().svd);
  idx const rows = n * n;
  auto matrix = uni20::reshape(std::move(theta), rows, rows);
  uni20::CudaMatrix<double> left(resources, rows, rows), right_adjoint(resources, rows, rows);
  uni20::CudaTensor<double, 1> values(resources, rows);
  uni20::linalg::singular_value_decomposition(values, left, right_adjoint, matrix);
  auto const s = uni20::to_host(values);
  auto const u = uni20::to_host(left);

  idx const k = static_cast<idx>(std::min<std::size_t>(D, static_cast<std::size_t>(rows)));
  host3 U(n, n, k);
  std::vector<double> S;
  double trace = 0.0, kept = 0.0;
  for (idx i = 0; i < rows; ++i)
    trace += static_cast<double>(s[i]);
  for (idx c = 0; c < k; ++c)
  {
    S.push_back(static_cast<double>(s[c]));
    kept += S.back();
    for (idx b = 0; b < n; ++b)
      for (idx a = 0; a < n; ++a)
        U[a, b, c] = static_cast<double>(u[a + n * b, c]);
  }
  double const gap = relative_gap(S.back(), k < rows, k < rows ? static_cast<double>(s[k]) : 0.0);
  return {.U = uni20::to_device(U, resources), .S = std::move(S), .discarded = 1.0 - kept / trace, .gap = gap};
}
#endif

} // namespace shim

// ---------------------------------------------------------------------------------------------
// the step: one text for both places
// ---------------------------------------------------------------------------------------------

// The same tensor with the two axes' names exchanged, T[u, l, r, d] -> [l, u, d, r].
template <class Tensor4> [[nodiscard]] auto exchange_axes(Tensor4 const& t)
{
  auto w = detail::wrap(t);
  return detail::permuted<1, 0, 3, 2>(w);
}

// The gram of the fused xm pair, theta[l1, l2, l1', l2'].  See gram_y in dense_hotrg.hpp for the picture.
template <class Tensor4> [[nodiscard]] auto gram_y(Tensor4 const& t)
{
  using Place = place_of<Tensor4>;
  using std::array;
  idx const m = t.extent(0), n = t.extent(1);
  reference::detail::require(t.extent(2) == n && t.extent(3) == m, "gram: opposite faces differ in dimension");

  auto site = detail::wrap(t);
  auto const t_ld_ur = detail::permuted<1, 3, 0, 2>(site); // [l, d, u, r]
  auto const t_lu_rd = detail::permuted<1, 0, 2, 3>(site); // [l, u, r, d]

  auto upper = detail::make_wrap<Place>(n, m, n, m); // A[l1, s, l1', s'], s the d face of the upper copy
  detail::contract_into(upper, t_ld_ur, t_ld_ur, array<axis_pair, 2>{{{2, 2}, {3, 3}}});
  auto lower = detail::make_wrap<Place>(n, m, n, m); // B[l2, s, l2', s'], s the u face of the lower copy
  detail::contract_into(lower, t_lu_rd, t_lu_rd, array<axis_pair, 2>{{{2, 2}, {3, 3}}});

  auto const a = detail::permuted<0, 2, 1, 3>(upper); // [l1, l1', s, s']
  auto const b = detail::permuted<1, 3, 0, 2>(lower); // [s, s', l2, l2']
  auto paired = detail::make_wrap<Place>(n, n, n, n); // [l1, l1', l2, l2']
  detail::contract_into(paired, a, b, array<axis_pair, 2>{{{2, 0}, {3, 1}}});
  return detail::permuted<0, 2, 1, 3>(paired); // [l1, l2, l1', l2']
}

// T'[u, k1, k2, d] = sum U[l1, l2, k1] Tu[u, l1, r1, s] Td[s, l2, r2, d] U[r1, r2, k2], in the order
// ((Tu U) Td) U: chi^6, chi^7, chi^6 in time.  The two chi^5 intermediates are five-leg tensors, and
// the scopes keep at most two of them alive at a time.
template <class Tensor4, class Tensor3>
[[nodiscard]] auto apply_isometry(Tensor4 const& upper, Tensor4 const& lower, Tensor3 const& iso)
{
  using Place = place_of<Tensor4>;
  using std::array;
  idx const mu = upper.extent(0), n = upper.extent(1), ms = upper.extent(3), md = lower.extent(3), k = iso.extent(2);
  reference::detail::require(upper.extent(2) == n && lower.extent(1) == n && lower.extent(2) == n &&
                                 lower.extent(0) == ms,
                             "apply_isometry: the two tensors do not stack");
  reference::detail::require(iso.extent(0) == n && iso.extent(1) == n,
                             "apply_isometry: the isometry does not fit the fused pair");

  auto second = [&] {
    auto const h1 = [&] {
      auto up = detail::wrap(upper);
      auto const tu = detail::permuted<0, 2, 3, 1>(up);       // [u, r1, s, l1]
      auto first = detail::make_wrap<Place>(mu, n, ms, n, k); // H1[u, r1, s, l2, k1]
      detail::contract_into(first, tu, iso, array<axis_pair, 1>{{{3, 0}}});
      return detail::permuted<0, 1, 4, 3, 2>(first); // [u, r1, k1, l2, s]
    }();
    auto low = detail::wrap(lower);
    auto const td = detail::permuted<1, 0, 2, 3>(low);     // [l2, s, r2, d]
    auto held = detail::make_wrap<Place>(mu, n, k, n, md); // H2[u, r1, k1, r2, d]
    detail::contract_into(held, h1, td, array<axis_pair, 2>{{{3, 0}, {4, 1}}});
    return held;
  }();
  auto const h2 = detail::permuted<2, 4, 0, 1, 3>(second); // [k1, d, u, r1, r2]
  auto third = detail::make_wrap<Place>(k, md, mu, k);     // [k1, d, u, k2]
  detail::contract_into(third, h2, iso, array<axis_pair, 2>{{{3, 0}, {4, 1}}});
  return detail::permuted<2, 0, 3, 1>(third); // [u, k1, k2, d]
}

template <class Place> struct step_result
{
    typename Place::template tensor<4> T; // the coarse tensor [ym, xm, xp, yp], divided by `norm`
    typename Place::template tensor<3> U; // the isometry [a, b, k]
    std::vector<double> S;                // the kept singular values of the gram, descending
    double norm;                          // the largest entry the coarse tensor was divided by
    double discarded;                     // the discarded fraction of the gram's trace
    double gap;                           // the relative gap at the cut
};

namespace detail
{

template <class Tensor4> [[nodiscard]] auto coarse_grain_y(Tensor4 const& t, std::size_t D, device_svd route)
{
  using Place = place_of<Tensor4>;
  idx const n = t.extent(1);
  auto cut = shim::cut_gram(Place{}, gram_y(t), n, D, route); // M1, M2
  auto T = apply_isometry(t, t, cut.U);
  double const norm = shim::largest_entry(Place{}, T); // M3
  reference::detail::require(norm > 0.0, "coarse_grain: the largest entry of the coarse tensor is not positive");
  uni20::transform_inplace(T, uni20::linalg::scale{1.0 / norm});
  return step_result<Place>{.T = std::move(T),
                            .U = std::move(cut.U),
                            .S = std::move(cut.S),
                            .norm = norm,
                            .discarded = cut.discarded,
                            .gap = cut.gap};
}

} // namespace detail

// One HOTRG step along `along` with a global top-D cut, for a host or a device tensor.
template <class Tensor4>
[[nodiscard]] auto coarse_grain(Tensor4 const& t, axis along, std::size_t D, device_svd route = device_svd::cusolver)
{
  if (along == axis::y) return detail::coarse_grain_y(t, D, route);
  auto result = detail::coarse_grain_y(exchange_axes(t), D, route);
  result.T = exchange_axes(result.T);
  return result;
}

// The ladder of dense_hotrg.hpp.  The tensor stays in its place from the first step to the last;
// what the record needs from it at each step goes through `shim`.
template <class Place>
[[nodiscard]] reference::flow_record flow(reference::tensor4 const& unit, std::size_t D, int steps,
                                          device_svd route = device_svd::cusolver,
                                          reference::flow_observer const& observe = {})
{
  reference::detail::require(steps >= 0 && steps <= reference::max_rounds,
                             "flow: the number of rounds is negative or above max_rounds, where the width 2^rounds "
                             "overflows");
  reference::flow_record record;
  record.trace_T_unit = reference::torus_trace(unit);
  auto T0 = shim::place(Place{}, unit); // M5
  long long size = 1;
  for (int step = 0; step < steps; ++step)
  {
    auto const start = std::chrono::steady_clock::now();
    seconds_spent const before = spent();
    auto along_y = coarse_grain(T0, axis::y, D, route);
    auto along_x = coarse_grain(along_y.T, axis::x, D, route);
    T0 = std::move(along_x.T);
    size *= 2;

    record.L.push_back(size);
    record.Ny.push_back(along_y.norm);
    record.Nx.push_back(along_x.norm);
    record.Norms.push_back(along_y.norm * along_y.norm * along_x.norm);
    record.discarded_y.push_back(along_y.discarded);
    record.discarded_x.push_back(along_x.discarded);
    record.gap_y.push_back(along_y.gap);
    record.gap_x.push_back(along_x.gap);
    record.trace_T.push_back(shim::torus_trace(Place{}, T0)); // M4
    record.lnz.push_back(reference::lnz_per_site(record.Norms, record.Norms.size(), record.trace_T.back()));
    Place::drain();
    record.seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    record.seconds_contract.push_back(spent().contract - before.contract);
    record.seconds_regroup.push_back(spent().regroup - before.regroup);
    record.seconds_svd.push_back(spent().svd - before.svd);
    if (observe) observe(record, shim::on_host(Place{}, T0)); // M5
  }
  record.T0 = shim::place(host_place{}, shim::on_host(Place{}, T0));
  return record;
}

} // namespace uni20::examples::dense_hotrg_shared
