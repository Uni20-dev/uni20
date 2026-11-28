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
// `contract` never packs an operand, so every contraction below is written as a matrix product
// -- the surviving group leading or trailing, the summed group adjacent on both operands --
// which is what reaches GEMM.  Every regrouping in between is a `permute<>` view materialized
// as a column-major tensor (`detail::permuted`).
//
// The step is written once for two places: host tensors of any real scalar type, and, in a build
// with the CUDA, cuBLAS and cuSOLVER backends, CUDA tensors of float or double (`host_place`,
// `device_place`).  Where a CUDA tensor has no counterpart of a host idiom there is a pair of
// overloads, and the device overload says what it stands in for:
//
//   * entry access (by design): the model tensor is built on the host and transferred, and the
//     flow's observer receives a host copy of the coarse tensor;
//   * the largest entry and the traces use `maximum_host` and `sum_host` of a `diagonal_view`;
//   * the gram is cut by a full cuSOLVER SVD (there is no device `truncated_svd`), and, since a
//     device matrix has no column slice, its leading columns are taken by a GEMM with the
//     leading columns of the identity.
//
// Scalar-generic real arithmetic with explicit working precision, unit cell 1x1.
// `ring_levels` reads the transfer-matrix spectrum of the coarse tensor back out,
// which the example compares against exact Ising values
// (ising_strip_reference.hpp).
#pragma once

#include <uni20/config.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/linalg/ops/randomized_nystrom.hpp>
#include <uni20/linalg/ops/randomized_svd.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/permute.hpp>
#include <uni20/tensor/reshape.hpp>
#include <uni20/tensor/tensor.hpp>

#if UNI20_BACKEND_CUDA && UNI20_BACKEND_CUBLAS && UNI20_BACKEND_CUSOLVER
#define UNI20_EXAMPLE_DENSE_HOTRG_DEVICE 1
#include <fmt/format.h>
#include <uni20/backend/cublas/execution.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/backend/cusolver/execution.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/linalg/ops/cuda_jacobi.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/diagonal.hpp>
#include <uni20/tensor/reductions.hpp>
#include <uni20/tensor/transform.hpp>
#else
#define UNI20_EXAMPLE_DENSE_HOTRG_DEVICE 0
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::examples::dense_hotrg
{

using idx = uni20::index_type;
using axis_pair = std::pair<std::size_t, std::size_t>;
template <uni20::Real R> using tensor3 = uni20::Tensor<R, 3>;
template <uni20::Real R> using tensor4 = uni20::Tensor<R, 4>;
template <uni20::Real R> using tensor5 = uni20::Tensor<R, 5>;

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

// How a step cuts the gram.  The gram is symmetric positive semidefinite, so its leading singular
// vectors are its leading eigenvectors. Randomized methods approximate their subspace.
//   svd   the leading singular vectors: `truncated_svd` on the host, the full cuSOLVER SVD on a device
//   eigh  the leading eigenvectors: `eigh` on the host (LAPACK, or MPLAPACK for the wide types) and on
//         a device (cuSOLVER syevd)
//   selected-eigh  only the leading k + 1 eigenpairs, the last for the gap: LAPACK syevr on the host,
//         cuSOLVER syevdx on a device
//   jacobi-eigh  the leading eigenvectors by cuSOLVER Jacobi (syevj); CUDA tensors only
//   jacobi-svd   the leading singular vectors by cuSOLVER Jacobi (gesvdj); CUDA tensors only
//   dc-eigh  the leading eigenvectors by LAPACK divide and conquer (syevd); host tensors only, since
//            eigh on a device already uses cuSOLVER's divide-and-conquer syevd
//   dc-svd   the leading singular vectors of a divide-and-conquer SVD (LAPACK gesdd); host tensors only
//   rsvd, nystrom  leading triplets or PSD eigenpairs from a Gaussian sketch on the host or device
enum class cut_method
{
  svd,
  eigh,
  selected_eigh,
  jacobi_eigh,
  jacobi_svd,
  dc_eigh,
  dc_svd,
  rsvd,
  nystrom
};

inline constexpr std::array all_cut_methods{cut_method::svd,         cut_method::eigh,       cut_method::selected_eigh,
                                            cut_method::jacobi_eigh, cut_method::jacobi_svd, cut_method::dc_eigh,
                                            cut_method::dc_svd,      cut_method::rsvd,       cut_method::nystrom};

struct sketch_options
{
    std::uint64_t seed = 1;
    idx oversampling = 8;
    idx power_iterations = 3;
};

// Location of this Gram in the physical flow, including when the x step exchanges tensor axes.
struct cut_position
{
    std::uint64_t round = 0;
    axis along = axis::y;
};

[[nodiscard]] constexpr bool uses_sketch(cut_method method)
{
  return method == cut_method::rsvd || method == cut_method::nystrom;
}

// Whether a method has only a CUDA implementation: the cuSOLVER Jacobi solvers have no host counterpart.
[[nodiscard]] constexpr bool cut_method_needs_device(cut_method method)
{
  return method == cut_method::jacobi_eigh || method == cut_method::jacobi_svd;
}

// Whether a method has only a host implementation: the explicit LAPACK divide-and-conquer drivers (#205).
[[nodiscard]] constexpr bool cut_method_needs_host(cut_method method)
{
  return method == cut_method::dc_eigh || method == cut_method::dc_svd;
}

// Whether the selected eigensolver (#204) covers a host scalar type: the raw LAPACK wrappers, which take
// float and double and, when configured, MPLAPACK binary128.  binary80 and MPFR have only the full solver.
template <uni20::Real R> inline constexpr bool selected_eigh_supported = uni20::LapackReal<R>;

// The divide-and-conquer drivers (#205) cover the same scalars; binary80 and MPFR decline them.
template <uni20::Real R> inline constexpr bool divide_and_conquer_supported = uni20::LapackReal<R>;

[[nodiscard]] inline char const* cut_method_name(cut_method method)
{
  switch (method)
  {
    case cut_method::svd:
      return "svd";
    case cut_method::eigh:
      return "eigh";
    case cut_method::selected_eigh:
      return "selected-eigh";
    case cut_method::jacobi_eigh:
      return "jacobi-eigh";
    case cut_method::jacobi_svd:
      return "jacobi-svd";
    case cut_method::dc_eigh:
      return "dc-eigh";
    case cut_method::dc_svd:
      return "dc-svd";
    case cut_method::rsvd:
      return "rsvd";
    case cut_method::nystrom:
      return "nystrom";
  }
  return "?";
}

// Where a run's time goes, accumulated over every call since the program started: the GEMM
// contractions, the regroupings (strided copies that materialize a permuted view), and the cut of
// the gram (its decomposition, and on a device the selection of the kept vectors).
struct seconds_spent
{
    double contract = 0.0, regroup = 0.0, cut = 0.0;
};

inline seconds_spent& spent()
{
  static seconds_spent totals;
  return totals;
}

// ---------------------------------------------------------------------------------------------
// the places where a step's tensors live
// ---------------------------------------------------------------------------------------------

// Host memory: any real scalar type, at working precision p.
struct host_place
{
    template <uni20::Real R, std::size_t Rank> using tensor = uni20::Tensor<R, Rank>;
    static constexpr char const* name = "host";

    // A tensor whose every element is written before it is read.
    template <uni20::Real R, class... Extents> [[nodiscard]] auto make(uni20::Precision p, Extents... extents) const
    {
      return tensor<R, sizeof...(Extents)>(uni20::uninitialized, static_cast<idx>(extents)..., p);
    }
};

#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
// One CUDA device: float or double.  The place of a tensor (`place_of`) is that tensor's device;
// a default-constructed place is the runtime's default device.
struct device_place
{
    template <uni20::Real R, std::size_t Rank> using tensor = uni20::CudaTensor<R, Rank>;
    static constexpr char const* name = "CUDA";

    uni20::cuda::DeviceResources* resources = &uni20::cuda::device_resources();

    template <uni20::Real R, class... Extents> [[nodiscard]] auto make(uni20::Precision, Extents... extents) const
    {
      return tensor<R, sizeof...(Extents)>(uni20::uninitialized, *resources, static_cast<idx>(extents)...);
    }
    template <uni20::Real R, std::size_t Rank> [[nodiscard]] auto upload(uni20::Tensor<R, Rank> const& t) const
    {
      return uni20::to_device(t, *resources);
    }
};
#endif

// Where a tensor lives: the host, or the device that holds its storage.
template <class Tensor> [[nodiscard]] auto place_of([[maybe_unused]] Tensor const& t)
{
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
  if constexpr (std::same_as<uni20::tensor_storage_policy_t<std::remove_cvref_t<Tensor>>, uni20::CudaStorage>)
    return device_place{&uni20::mdspec_of(t).data_descriptor().buffer().resources()};
  else
    return host_place{};
#else
  return host_place{};
#endif
}

template <class Tensor> using place_type = decltype(place_of(std::declval<Tensor const&>()));

// Wait until the work that writes `t` has finished, so that a timer sees it.  Host work has
// finished already.  A CUDA tensor's buffer waits for the submitted operations that involve it,
// and for nothing else submitted to the device.
template <class Tensor> void wait_for([[maybe_unused]] Tensor const& t)
{
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
  if constexpr (std::same_as<uni20::tensor_storage_policy_t<std::remove_cvref_t<Tensor>>, uni20::CudaStorage>)
    uni20::mdspec_of(t).data_descriptor().buffer().synchronize();
#endif
}

namespace detail
{

// Wall time spent in a scope.  A timed scope that queues device work ends with `wait_for` on the
// tensor that work writes; each scope's inputs were waited for by the scope that wrote them.
struct lap
{
    double& slot;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    explicit lap(double& target) : slot(target) {}
    lap(lap const&) = delete;
    lap& operator=(lap const&) = delete;
    ~lap() { slot += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};

// The tensor with its axes regrouped, as an owning column-major tensor in the same place, at
// working precision p: output axis j is source axis Axis[j].  The copy writes every element.
template <std::size_t... Axis, class Source> [[nodiscard]] auto permuted(Source const& t, uni20::Precision p)
{
  auto const place = place_of(t);
  lap const timer(spent().regroup);
  auto result = place.template make<typename Source::value_type>(p, t.extent(Axis)...);
  uni20::copy(result, uni20::permute<Axis...>(t));
  wait_for(result);
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

// sum_{i, j} theta[i, j, i, j]: the trace of the gram as a matrix [(l1, l2); (l1', l2')].
template <uni20::Real R> [[nodiscard]] inline R gram_trace(tensor4<R> const& theta)
{
  auto const p = uni20::precision_of(theta);
  R trace = uni20::scalar_at_precision<R>(0, p);
  for (idx i = 0; i < theta.extent(0); ++i)
    for (idx j = 0; j < theta.extent(1); ++j)
      trace += R(theta[i, j, i, j]);
  return trace;
}

#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
// The same four pieces for a CUDA tensor, which has no element access.  Each reads the tensor in
// place and returns one host scalar.
template <uni20::Real R> [[nodiscard]] inline R max_element(uni20::CudaTensor<R, 4> const& t)
{
  return std::max(R{0}, uni20::maximum_host(t));
}

template <uni20::Real R> inline void scale(uni20::CudaTensor<R, 4>& t, R factor)
{
  uni20::transform_inplace(t, uni20::linalg::scale{factor});
}

template <uni20::Real R> [[nodiscard]] inline R torus_trace(uni20::CudaTensor<R, 4> const& t)
{
  detail::require(t.extent(0) == t.extent(3) && t.extent(1) == t.extent(2),
                  "torus_trace: opposite faces differ in dimension");
  return uni20::sum_host(uni20::diagonal_view<0, 3, 1, 2>(t));
}

template <uni20::Real R> [[nodiscard]] inline R gram_trace(uni20::CudaTensor<R, 4> const& theta)
{
  return uni20::sum_host(uni20::diagonal_view<0, 2, 1, 3>(theta));
}
#endif

// The same tensor with the two axes' names exchanged, T[u, l, r, d] -> [l, u, d, r].
template <class Tensor4> [[nodiscard]] inline auto exchange_axes(Tensor4 const& t)
{
  auto const p = uni20::precision_of(t);
  return detail::permuted<1, 0, 3, 2>(t, p);
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
template <class Tensor4> [[nodiscard]] inline auto gram_y(Tensor4 const& t)
{
  using R = typename Tensor4::value_type;
  auto const place = place_of(t);
  auto const p = uni20::precision_of(t);
  idx const m = t.extent(0), n = t.extent(1);
  detail::require(t.extent(2) == n && t.extent(3) == m, "gram: opposite faces differ in dimension");
  using std::array;

  auto const t_ld_ur = detail::permuted<1, 3, 0, 2>(t, p); // [l, d, u, r]
  auto const t_lu_rd = detail::permuted<1, 0, 2, 3>(t, p); // [l, u, r, d]

  // Each contraction output is written by a beta = 0 contraction before it is read.
  auto upper = place.template make<R>(p, n, m, n, m); // A[l1, s, l1', s'], s the d face of the upper copy
  {
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(upper, uni20::scalar_at_precision<R>(1, p), t_ld_ur, t_ld_ur,
                            array<axis_pair, 2>{{{2, 2}, {3, 3}}}, uni20::scalar_at_precision<R>(0, p), p);
    wait_for(upper);
  }
  auto lower = place.template make<R>(p, n, m, n, m); // B[l2, s, l2', s'], s the u face of the lower copy
  {
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(lower, uni20::scalar_at_precision<R>(1, p), t_lu_rd, t_lu_rd,
                            array<axis_pair, 2>{{{2, 2}, {3, 3}}}, uni20::scalar_at_precision<R>(0, p), p);
    wait_for(lower);
  }
  auto const a = detail::permuted<0, 2, 1, 3>(upper, p); // [l1, l1', s, s']
  auto const b = detail::permuted<1, 3, 0, 2>(lower, p); // [s, s', l2, l2']

  auto paired = place.template make<R>(p, n, n, n, n); // [l1, l1', l2, l2']
  {
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(paired, uni20::scalar_at_precision<R>(1, p), a, b, array<axis_pair, 2>{{{2, 0}, {3, 1}}},
                            uni20::scalar_at_precision<R>(0, p), p);
    wait_for(paired);
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
template <class Tensor4, class Tensor3>
[[nodiscard]] inline auto apply_isometry(Tensor4 const& upper, Tensor4 const& lower, Tensor3 const& iso)
{
  using R = typename Tensor4::value_type;
  auto const place = place_of(upper);
  auto const p = uni20::precision_of(upper);
  idx const mu = upper.extent(0), n = upper.extent(1), ms = upper.extent(3), md = lower.extent(3), k = iso.extent(2);
  detail::require(upper.extent(2) == n && lower.extent(1) == n && lower.extent(2) == n && lower.extent(0) == ms,
                  "apply_isometry: the two tensors do not stack");
  detail::require(iso.extent(0) == n && iso.extent(1) == n, "apply_isometry: the isometry does not fit the fused pair");
  using std::array;

  auto const tu = detail::permuted<0, 2, 3, 1>(upper, p); // [u, r1, s, l1]
  auto const td = detail::permuted<1, 0, 2, 3>(lower, p); // [l2, s, r2, d]

  // H2[u, r1, k1, r2, d] = sum_{l2, s} H1 Td.  The scopes keep at most two chi^5 buffers alive
  // at a time.  Each contraction output is written by a beta = 0 contraction before it is read.
  auto const second = [&] {
    // H1[u, r1, s, l2, k1] = sum_{l1} Tu U, regrouped as [u, r1, k1, l2, s]
    auto const h1 = [&] {
      auto first = place.template make<R>(p, mu, n, ms, n, k);
      {
        detail::lap const timer(spent().contract);
        uni20::linalg::contract(first, uni20::scalar_at_precision<R>(1, p), tu, iso, array<axis_pair, 1>{{{3, 0}}},
                                uni20::scalar_at_precision<R>(0, p), p);
        wait_for(first);
      }
      return detail::permuted<0, 1, 4, 3, 2>(first, p);
    }();
    auto held = place.template make<R>(p, mu, n, k, n, md);
    {
      detail::lap const timer(spent().contract);
      uni20::linalg::contract(held, uni20::scalar_at_precision<R>(1, p), h1, td, array<axis_pair, 2>{{{3, 0}, {4, 1}}},
                              uni20::scalar_at_precision<R>(0, p), p);
      wait_for(held);
    }
    return held;
  }();
  auto const h2 = detail::permuted<2, 4, 0, 1, 3>(second, p); // [k1, d, u, r1, r2]

  // [k1, d, u, k2] = sum_{r1, r2} H2 U
  auto third = place.template make<R>(p, k, md, mu, k);
  {
    detail::lap const timer(spent().contract);
    uni20::linalg::contract(third, uni20::scalar_at_precision<R>(1, p), h2, iso, array<axis_pair, 2>{{{3, 0}, {4, 1}}},
                            uni20::scalar_at_precision<R>(0, p), p);
    wait_for(third);
  }
  return detail::permuted<2, 0, 3, 1>(third, p); // [u, k1, k2, d]
}

template <uni20::Real R, class Place = host_place> struct step_result
{
    typename Place::template tensor<R, 4> T; // the coarse tensor [ym, xm, xp, yp], divided by `norm`
    typename Place::template tensor<R, 3> U; // the isometry [a, b, k]: a the upper (left) copy's leg, b the lower one's
    std::vector<R> S = {};                   // the kept singular values of the gram, descending
    R norm = R{0};                           // the largest entry the coarse tensor was divided by
    R discarded = R{0};                      // 1 - sum(S) / tr(gram): the discarded fraction of the gram's trace
    R gap = R{0}; // (smallest kept - largest discarded) / smallest kept singular value; 1 if nothing was
                  // discarded, 0 if the smallest kept value is zero.  Near 0 the cut falls inside a
                  // degenerate cluster and the kept subspace is the solver's choice, not the model's.
};

namespace detail
{

// The leading singular vectors of the gram theta[l1, l2, l1', l2'] as an isometry U[l1, l2, k],
// with what the cut keeps and discards.
template <uni20::Real R, class Place> struct gram_cut
{
    typename Place::template tensor<R, 3> U;
    std::vector<R> S;
    R discarded;
    R gap;
};

// The cut from the gram's eigensystem, ascending: the last k eigenpairs, in descending order.
template <uni20::Real R, class Values>
void cut_from_eigenvalues(R& gap, std::vector<R>& S, R& discarded, Values const& values, idx rows, idx k, R trace)
{
  R kept = R{0};
  for (idx c = 0; c < k; ++c)
  {
    S.push_back(R(values[rows - 1 - c]));
    kept += S.back();
  }
  discarded = 1 - kept / trace;
  if (k == rows)
    gap = R{1}; // nothing was discarded
  else if (S.back() > 0)
    gap = (S.back() - R(values[rows - 1 - k])) / S.back();
  else
    gap = R{0}; // the cut falls inside the null space
}

// Descending values include the first omitted value whenever the cut truncates the Gram.
template <uni20::Real R, class Place, class Values>
void record_descending_cut(gram_cut<R, Place>& cut, Values const& values, idx rows, idx k, R trace)
{
  auto const p = uni20::precision_of(trace);
  R kept = uni20::scalar_at_precision<R>(0, p);
  cut.S.reserve(k);
  for (idx i = 0; i < k; ++i)
  {
    cut.S.push_back(R(values[i]));
    kept += cut.S.back();
  }
  cut.discarded = 1 - kept / trace;
  if (k == rows)
    cut.gap = uni20::scalar_at_precision<R>(1, p);
  else if (cut.S.back() > 0)
    cut.gap = (cut.S.back() - R(values[k])) / cut.S.back();
  else
    cut.gap = uni20::scalar_at_precision<R>(0, p);
}

// Both randomized solvers return descending values and owning column-major vectors in the input's place.
template <class Tensor4>
[[nodiscard]] auto randomized_cut(Tensor4 const& theta, idx k, typename Tensor4::value_type trace, cut_method method,
                                  sketch_options const& sketch, cut_position position)
{
  using R = typename Tensor4::value_type;
  using Place = place_type<Tensor4>;
  idx const n = theta.extent(0), rows = n * n;
  uni20::linalg::RandomizedRangeOptions options{
      .rank = k < rows ? k + 1 : k, .oversampling = sketch.oversampling, .power_iterations = sketch.power_iterations};
  auto const round = random::child(random::root(sketch.seed), position.round);
  auto const identity = random::child(round, static_cast<std::uint64_t>(position.along));
  auto const matrix = uni20::reshape_view(theta, rows, rows);
  lap const timer(spent().cut);
  auto finish = [&](auto vectors, auto const& values) {
    // Keep the column-major prefix in place. The allocation owns at most one extra column for the gap.
    using V = decltype(vectors);
    static_assert(std::same_as<typename V::layout_type, uni20::ColumnMajor>);
    vectors.replace_mapping(typename V::mapping_type{typename V::extents_type{rows, k}});
    gram_cut<R, Place> cut{.U = uni20::reshape(std::move(vectors), n, n, k), .S = {}, .discarded = R{0}, .gap = R{0}};
    wait_for(cut.U);
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
    if constexpr (std::same_as<Place, device_place>)
      record_descending_cut(cut, uni20::to_host(values), rows, k, trace);
    else
#endif
      record_descending_cut(cut, values, rows, k, trace);
    return cut;
  };
  if (method == cut_method::rsvd)
  {
    auto result = uni20::linalg::randomized_truncated_svd(matrix, options, identity);
    return finish(std::move(result.left_singular_vectors), result.singular_values);
  }
  auto result = uni20::linalg::randomized_nystrom(matrix, options, identity);
  return finish(std::move(result.eigenvectors), result.eigenvalues);
}

template <uni20::Real R>
[[nodiscard]] inline gram_cut<R, host_place> cut_gram(tensor4<R> const& theta, std::size_t D, cut_method method,
                                                      sketch_options const& sketch = {}, cut_position position = {})
{
  auto const p = uni20::precision_of(theta);
  idx const n = theta.extent(0);
  R const trace = gram_trace(theta);

  std::size_t const rows = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  if (cut_method_needs_device(method))
    throw std::invalid_argument(std::string("dense_hotrg: the ") + cut_method_name(method) +
                                " cut runs only on CUDA tensors");
  idx const k = static_cast<idx>(std::min(D, rows));
  if (uses_sketch(method))
  {
    if constexpr (uni20::linalg::detail::randomized::SketchReal<R>)
      return randomized_cut(theta, k, trace, method, sketch, position);
    else
      throw std::invalid_argument("dense_hotrg: randomized Gram cuts require a QR and SVD provider at this precision");
  }
  // Ascending eigenpairs, `count` of them: U is the last k eigenvectors in descending order.
  auto const from_ascending = [&](auto const& decomposition, idx count) {
    gram_cut<R, host_place> result{
        .U = tensor3<R>(uni20::uninitialized, n, n, k, p), .S = {}, .discarded = R{0}, .gap = R{0}};
    for (idx c = 0; c < k; ++c)
      for (idx b = 0; b < n; ++b)
        for (idx a = 0; a < n; ++a)
          result.U[a, b, c] = R(decomposition.eigenvectors[a + n * b, count - 1 - c]);
    cut_from_eigenvalues(result.gap, result.S, result.discarded, decomposition.eigenvalues, count, k, trace);
    return result;
  };
  // A complete SVD, singular values descending: U is the leading k left vectors.
  auto const from_complete_svd = [&](auto const& decomposition) {
    gram_cut<R, host_place> result{
        .U = tensor3<R>(uni20::uninitialized, n, n, k, p), .S = {}, .discarded = R{0}, .gap = R{0}};
    for (idx c = 0; c < k; ++c)
      for (idx b = 0; b < n; ++b)
        for (idx a = 0; a < n; ++a)
          result.U[a, b, c] = R(decomposition.left_singular_vectors[a + n * b, c]);
    record_descending_cut(result, decomposition.singular_values, n * n, k, trace);
    return result;
  };

  if (method == cut_method::selected_eigh)
  {
    if constexpr (!selected_eigh_supported<R>)
      throw std::invalid_argument(
          "dense_hotrg: the selected-eigh cut needs a LAPACK scalar (float, double, MPLAPACK binary128); use eigh");
    else
    {
      idx const r = n * n;
      idx const first = std::max<idx>(r - k - 1, 0); // one more than kept, for the gap
      auto const decomposition = [&] {
        lap const timer(spent().cut);
        return uni20::linalg::eigh(uni20::reshape_view(theta, r, r), uni20::linalg::EigenvalueIndexRange{first, r},
                                   uni20::linalg::MatrixTriangle::Upper, p); // ascending
      }();
      return from_ascending(decomposition, r - first);
    }
  }
  if (method == cut_method::dc_eigh || method == cut_method::dc_svd)
  {
    if constexpr (!divide_and_conquer_supported<R>)
      throw std::invalid_argument(std::string("dense_hotrg: the ") + cut_method_name(method) +
                                  " cut needs a LAPACK scalar (float, double, MPLAPACK binary128); use eigh or svd");
    else
    {
      auto const matrix = uni20::reshape_view(theta, n * n, n * n);
      if (method == cut_method::dc_eigh)
      {
        auto const decomposition = [&] {
          lap const timer(spent().cut);
          return uni20::linalg::eigh(matrix, uni20::linalg::EighDriver::DivideAndConquer,
                                     uni20::linalg::MatrixTriangle::Upper, p); // ascending
        }();
        return from_ascending(decomposition, n * n);
      }
      auto const decomposition = [&] {
        lap const timer(spent().cut);
        return uni20::linalg::svd(matrix,
                                  uni20::linalg::SvdOptions{.driver = uni20::linalg::SvdDriver::DivideAndConquer},
                                  p); // descending
      }();
      return from_complete_svd(decomposition);
    }
  }
  if (method == cut_method::eigh)
  {
    auto const decomposition = [&] {
      lap const timer(spent().cut);
      return uni20::linalg::eigh(uni20::reshape_view(theta, n * n, n * n), p); // ascending
    }();
    return from_ascending(decomposition, n * n);
  }
  auto decomposition = [&] {
    lap const timer(spent().cut);
    return uni20::linalg::truncated_svd(
        uni20::reshape_view(theta, n * n, n * n),
        uni20::linalg::SvdTruncationPolicy<R>{.maximum_retained_extent = std::min(D, rows)}, p);
  }();
  idx const kept_extent = static_cast<idx>(decomposition.singular_values.extent(0));

  gram_cut<R, host_place> result{
      .U = tensor3<R>(uni20::uninitialized, n, n, kept_extent, p), .S = {}, .discarded = R{0}, .gap = R{0}};
  uni20::copy(result.U, uni20::reshape_view(decomposition.left_singular_vectors, n, n, kept_extent));
  R kept = uni20::scalar_at_precision<R>(0, p);
  for (idx i = 0; i < kept_extent; ++i)
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
  return result;
}

#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
// A CUDA tensor has no `truncated_svd`, and no view of a matrix's leading columns.  The full SVD
// runs on the device through cuSOLVER `gesvd`; its singular values (n^2 scalars) are read on the
// host to report the cut; and U is the left factor times the leading k columns of the identity,
// one GEMM on the device, so the factor never leaves the device.  The SVD computes the right
// factor and all n^2 vectors where k are kept.
//
// The eigensolvers (syevd, syevdx for selected-eigh, syevj for jacobi-eigh) give ascending eigenpairs
// on the device, and U is the eigenvector matrix times the last k columns of the identity in reverse
// order, so it is descending.  jacobi-svd (gesvdj) gives the reduced SVD, used as the full one is.
template <uni20::Real R>
[[nodiscard]] inline gram_cut<R, device_place> cut_gram(uni20::CudaTensor<R, 4> theta, std::size_t D, cut_method method,
                                                        sketch_options const& sketch = {}, cut_position position = {})
{
  auto const p = uni20::precision_of(theta);
  auto const place = place_of(theta);
  auto& resources = *place.resources;
  idx const n = theta.extent(0), rows = n * n;
  idx const k = static_cast<idx>(std::min<std::size_t>(D, static_cast<std::size_t>(rows)));
  R const trace = gram_trace(theta);
  if (cut_method_needs_host(method))
    throw std::invalid_argument(fmt::format("dense_hotrg: the {} cut runs only on host tensors; on CUDA, eigh "
                                            "already uses cuSOLVER's divide-and-conquer syevd",
                                            cut_method_name(method)));

  if (uses_sketch(method)) return randomized_cut(theta, k, trace, method, sketch, position);

  // U is the solver's vectors times a selector, columns of the identity, in one GEMM on the device.
  auto const keep_columns = [&](auto& U, auto const& vectors, idx count, bool trailing) {
    uni20::Tensor<R, 2> selector(count, k, p); // zero-initialized
    for (idx c = 0; c < k; ++c)
      selector[trailing ? count - 1 - c : c, c] = R{1};  // trailing: ascending vectors, kept in reverse
    auto const device_selector = place.upload(selector); // alive until the GEMM has finished
    uni20::linalg::contract(uni20::reshape_view(U, rows, k), R{1}, vectors, device_selector,
                            std::array<axis_pair, 1>{{{1, 0}}}, R{0}, p);
    wait_for(U);
  };
  // A Jacobi solver that reaches its sweep limit returns approximate factors; a cut from those would be
  // silently wrong, so stop instead.
  auto const converged = [&](auto result) {
    auto const& c = result.convergence;
    if (!c.converged)
      throw std::runtime_error(fmt::format("dense_hotrg: the {} cut did not converge in {} sweeps (residual {:.3e})",
                                           cut_method_name(method), c.sweeps, c.residual));
    return std::move(result.factors);
  };

  if (method == cut_method::selected_eigh || method == cut_method::eigh || method == cut_method::jacobi_eigh)
  {
    // Ascending eigenpairs; selected-eigh asks for one more than kept, for the gap.
    idx const first = method == cut_method::selected_eigh ? std::max<idx>(rows - k - 1, 0) : 0;
    idx const count = rows - first;
    auto U = place.template make<R>(p, n, n, k);
    auto const values = [&] {
      lap const timer(spent().cut);
      auto const matrix = uni20::reshape_view(theta, rows, rows);
      if (method == cut_method::jacobi_eigh)
      {
        auto const decomposition = converged(uni20::linalg::eigh_jacobi(matrix));
        keep_columns(U, decomposition.eigenvectors, count, true);
        return uni20::to_host(decomposition.eigenvalues);
      }
      if (method == cut_method::selected_eigh)
      {
        auto const decomposition = uni20::linalg::eigh(matrix, uni20::linalg::EigenvalueIndexRange{first, rows});
        keep_columns(U, decomposition.eigenvectors, count, true);
        return uni20::to_host(decomposition.eigenvalues);
      }
      auto const decomposition = uni20::linalg::eigh(matrix);
      keep_columns(U, decomposition.eigenvectors, count, true);
      return uni20::to_host(decomposition.eigenvalues);
    }();
    gram_cut<R, device_place> result{.U = std::move(U), .S = {}, .discarded = R{0}, .gap = R{0}};
    cut_from_eigenvalues(result.gap, result.S, result.discarded, values, count, k, trace);
    return result;
  }

  // Descending singular values.
  auto U = place.template make<R>(p, n, n, k);
  auto const s = [&] {
    lap const timer(spent().cut);
    if (method == cut_method::jacobi_svd)
    {
      auto const decomposition = converged(uni20::linalg::svd_jacobi(uni20::reshape_view(theta, rows, rows)));
      keep_columns(U, decomposition.left_singular_vectors, rows, false);
      return uni20::to_host(decomposition.singular_values);
    }
    uni20::CudaMatrix<R> left(uni20::uninitialized, resources, rows, rows);
    uni20::CudaMatrix<R> right_adjoint(uni20::uninitialized, resources, rows, rows);
    uni20::CudaTensor<R, 1> values(uni20::uninitialized, resources, rows);
    auto matrix = uni20::reshape(std::move(theta), rows, rows);
    uni20::linalg::singular_value_decomposition(values, left, right_adjoint, matrix);
    keep_columns(U, left, rows, false);
    return uni20::to_host(values);
  }();

  gram_cut<R, device_place> result{.U = std::move(U), .S = {}, .discarded = R{0}, .gap = R{0}};
  record_descending_cut(result, s, rows, k, trace);
  return result;
}
#endif

template <class Tensor4>
[[nodiscard]] inline auto coarse_grain_y(Tensor4 const& t, std::size_t D, cut_method method,
                                         sketch_options const& sketch, cut_position position)
{
  using R = typename Tensor4::value_type;
  auto cut = cut_gram(gram_y(t), D, method, sketch, position);
  step_result<R, place_type<Tensor4>> result{.T = apply_isometry(t, t, cut.U),
                                             .U = std::move(cut.U),
                                             .S = std::move(cut.S),
                                             .discarded = cut.discarded,
                                             .gap = cut.gap};
  result.norm = max_element(result.T);
  require(result.norm > 0, "coarse_grain: the largest entry of the coarse tensor is not positive");
  scale(result.T, R(1 / result.norm));
  return result;
}

} // namespace detail

// The gram of the fused low-face pair of the OTHER axis (xm for a y step, ym for an x step).
template <class Tensor4> [[nodiscard]] inline auto gram(Tensor4 const& t, axis along)
{
  return along == axis::y ? gram_y(t) : gram_y(exchange_axes(t));
}

// One HOTRG step along `along` with a global top-D cut: gram -> its leading vectors (`method`) ->
// the update network with U on both fused faces -> divide by the largest entry.  For the x step
// exchange the names ym <-> xm, yp <-> xp in every picture.  The tensor may be on the host or, in a
// CUDA build, on the device; the result is in the same place.
template <class Tensor4>
[[nodiscard]] inline auto coarse_grain(Tensor4 const& t, axis along, std::size_t D, cut_method method = cut_method::svd,
                                       sketch_options const& sketch = {}, std::uint64_t round = 0)
{
  if (along == axis::y) return detail::coarse_grain_y(t, D, method, sketch, {round, along});
  auto result = detail::coarse_grain_y(exchange_axes(t), D, method, sketch, {round, along});
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
    std::vector<double> seconds = {}, seconds_contract = {}, seconds_regroup = {}, seconds_cut = {};
    R initial_log_scale;
    R unit_lnz;
    tensor4<R> T0;
};

/// \brief Coarse-grain normalized tensors, retaining removed scales as logarithms.
/// \details One round doubles each side. No lattice extent or normalization product is formed.
///          The finite input tensor is normalized before its first Gram matrix; its
///          periodic traces must be positive for the partition-function logarithm.
///          The observer receives the current record and its normalized coarse tensor.
///          `Place` is where the coarse tensors live (host_place, or device_place in a CUDA build).
///          The model tensor is normalized on the host and placed once; the record's `T0`,
///          and the tensor given to the observer, are on the host.
template <class Place = host_place, uni20::Real R, class Observer = std::nullptr_t>
[[nodiscard]] flow_record<R> flow(scaled_tensor<R> unit, std::size_t D, int steps, Observer observe = nullptr,
                                  cut_method method = cut_method::svd, sketch_options const& sketch = {})
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
  [[maybe_unused]] Place const place{}; // where a host input is placed
  auto rounds = [&](auto& T) {
    for (int step = 0; step < steps; ++step)
    {
      auto const start = std::chrono::steady_clock::now();
      seconds_spent const before = spent();
      auto along_y = coarse_grain(T, axis::y, D, method, sketch, step);
      auto along_x = coarse_grain(along_y.T, axis::x, D, method, sketch, step);
      T = std::move(along_x.T);
      record.log2_width.push_back(step + 1);
      record.log_norm_y.push_back(uni20::math::log(along_y.norm));
      record.log_norm_x.push_back(uni20::math::log(along_x.norm));
      weight /= 4;
      sum += weight * (2 * record.log_norm_y.back() + record.log_norm_x.back());
      record.lnz.push_back(sum + weight * uni20::math::log(torus_trace(T)));
      record.discarded_y.push_back(along_y.discarded);
      record.discarded_x.push_back(along_x.discarded);
      record.gap_y.push_back(along_y.gap);
      record.gap_x.push_back(along_x.gap);
      wait_for(T);
      record.seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
      record.seconds_contract.push_back(spent().contract - before.contract);
      record.seconds_regroup.push_back(spent().regroup - before.regroup);
      record.seconds_cut.push_back(spent().cut - before.cut);
      if constexpr (!std::same_as<Observer, std::nullptr_t>)
      {
        if constexpr (std::same_as<Place, host_place>)
          observe(record, T);
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
        else
        {
          // The record's T0 is the latest tensor in both places when the observer runs.
          record.T0 = uni20::to_host(T);
          observe(record, record.T0);
        }
#endif
      }
    }
  };
  if constexpr (std::same_as<Place, host_place>) rounds(record.T0);
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
  else
  {
    auto T = place.upload(record.T0);
    rounds(T);
    // With an observer, every round has already downloaded T into T0.
    if constexpr (std::same_as<Observer, std::nullptr_t>) record.T0 = uni20::to_host(T);
  }
#endif
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
