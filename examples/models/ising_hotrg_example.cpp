// ising_hotrg_example.cpp -- a dense 2D HOTRG of the square-lattice Ising model.
//
// Without arguments it runs one flow at the critical temperature and prints, round by round,
// the free energy against Onsager's and the two lowest scaling dimensions read from the coarse
// tensor's transfer matrix against the exactly solved strip of the same width.
//
//   ising_hotrg_example [--bond=D] [--rounds=N] [--temperature=T/Tc] [--execution=cpu|cuda] [--check]
//
// --execution=cuda runs the same step on CUDA tensors (fp32 or fp64), in a build with the CUDA,
// cuBLAS and cuSOLVER backends.
//
// With --check it verifies the implementation instead and exits non-zero on any failure:
//
//   A  the kernel against the DEFINITION: gram, update network and one step along both axes, by
//      brute-force loops on a small random tensor with unequal leg dimensions
//   B  untruncated flows against partition functions enumerated over spin configurations
//   C  the q = 3 clock model at K is the 3-state Potts model at 3K/2, ln Z / N shifted by K
//   D  Onsager's free energy
//   E  every contraction of a round reaches GEMM
//   F  the coarse tensor's transfer matrix against the exactly solved Ising strip
//   G  the same reading along a truncated flow, width by width
//   H  inputs at the edge
//   I  with --execution=cuda: the step on CUDA tensors against the host, the gram and the update
//      network entry by entry, one step along each axis, and two flows.  Coarse tensors are not
//      compared entry by entry after a decomposition: two SVD codes may return a singular vector
//      with either sign, which changes the tensor but nothing read from it.  With
//      --second-cuda-device=N it also checks that a step on a tensor held by device N, which is
//      not the default, keeps its results there.
#include "dense_hotrg.hpp"
#include "ising_hotrg_output.hpp"
#include "ising_strip_reference.hpp"

#include <uni20/linalg/dispatch_diagnostics.hpp>

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string_view>

using namespace uni20::examples::dense_hotrg;

namespace
{
namespace p = uni20::presentation;
namespace d = uni20::display;
auto const text = p::style("");

using tensor3 = uni20::examples::dense_hotrg::tensor3<double>;
using tensor4 = uni20::examples::dense_hotrg::tensor4<double>;
using step_result = uni20::examples::dense_hotrg::step_result<double>;
using flow_record = uni20::examples::dense_hotrg::flow_record<double>;

int failures = 0;

void report(bool ok, std::string const& what)
{
  d::emit(d::status_cell(ok ? p::semantic_glyph::success : p::semantic_glyph::failure, "{}", what), d::stream::out);
  if (!ok) ++failures;
}

// ----------------------------------------------------------------------------------------------
// shared by the flow and by the checks
// ----------------------------------------------------------------------------------------------

constexpr std::size_t no_truncation = std::size_t{1} << 20;

reference::strip const* exact_strip(long long sites, double over_tc)
{
  for (auto const& candidate : reference::strips)
    if (candidate.sites == sites && static_cast<double>(candidate.numerator) / candidate.denominator == over_tc)
      return &candidate;
  return nullptr;
}

// One HOTRG round: the y step, then the x step.
tensor4 one_round(tensor4 const& t, std::size_t D)
{
  step_result const along_y = coarse_grain(t, axis::y, D);
  return coarse_grain(along_y.T, axis::x, D).T;
}

// How well X_i can be resolved at all.  The transfer matrix holds lambda_i / lambda_0 =
// exp(-2 pi X_i), and an eigensolver returns every eigenvalue with an absolute error of order
// eps lambda_0, so the relative error of lambda_i -- and with it the error of X_i -- grows as
// exp(+2 pi X_i).  A reading is correct when it sits inside this, not inside a fixed epsilon.
double resolution_of(double x)
{
  constexpr double eps = std::numeric_limits<double>::epsilon();
  return std::max(1e-14, 50.0 * eps * std::exp(2.0 * std::numbers::pi * x) / (2.0 * std::numbers::pi));
}

// Compares `count` levels and returns the largest deviation as a fraction of what is resolvable.
double worst_relative_to_resolution(std::vector<double> const& X, std::span<std::string_view const> expected,
                                    std::size_t count, double* worst_absolute = nullptr)
{
  double worst = 0.0, absolute = 0.0;
  for (std::size_t i = 0; i < count; ++i)
  {
    double const gap = std::abs(X[i] - uni20::parse_real<double>(expected[i]));
    absolute = std::max(absolute, gap);
    worst = std::max(worst, gap / resolution_of(uni20::parse_real<double>(expected[i])));
  }
  if (worst_absolute) *worst_absolute = absolute;
  return worst;
}

// ----------------------------------------------------------------------------------------------
// the flow
// ----------------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------------
// A  the kernel against the definition
// ----------------------------------------------------------------------------------------------

// Entries in (-0.4, 1): generic, no symmetry of any kind, the largest entry positive.
struct generator
{
    std::uint64_t state;
    double next()
    {
      this->state = this->state * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<double>(this->state >> 11) / 9007199254740992.0 * 1.4 - 0.4;
    }
};

tensor4 random4(idx e0, idx e1, idx e2, idx e3, std::uint64_t seed)
{
  tensor4 t(e0, e1, e2, e3);
  generator g{seed};
  for (auto& value : t.storage())
    value = g.next();
  return t;
}

double at(tensor4 const& t, idx a, idx b, idx c, idx d) { return static_cast<double>(t[a, b, c, d]); }
double at(tensor3 const& t, idx a, idx b, idx c) { return static_cast<double>(t[a, b, c]); }

// theta[l1, l2, e1, e2] of the y step, straight from the picture.
double gram_y_entry(tensor4 const& t, idx l1, idx l2, idx e1, idx e2)
{
  idx const m = t.extent(0), n = t.extent(1);
  double sum = 0.0;
  for (idx up = 0; up < m; ++up)
    for (idx r1 = 0; r1 < n; ++r1)
      for (idx r2 = 0; r2 < n; ++r2)
        for (idx down = 0; down < m; ++down)
        {
          double ket = 0.0, bra = 0.0;
          for (idx s = 0; s < m; ++s)
          {
            ket += at(t, up, l1, r1, s) * at(t, s, l2, r2, down);
            bra += at(t, up, e1, r1, s) * at(t, s, e2, r2, down);
          }
          sum += ket * bra;
        }
  return sum;
}

// theta[u1, u2, e1, e2] of the x step: two copies side by side on the xp -- xm bond.
double gram_x_entry(tensor4 const& t, idx u1, idx u2, idx e1, idx e2)
{
  idx const m = t.extent(0), n = t.extent(1);
  double sum = 0.0;
  for (idx left = 0; left < n; ++left)
    for (idx d1 = 0; d1 < m; ++d1)
      for (idx right = 0; right < n; ++right)
        for (idx d2 = 0; d2 < m; ++d2)
        {
          double ket = 0.0, bra = 0.0;
          for (idx s = 0; s < n; ++s)
          {
            ket += at(t, u1, left, s, d1) * at(t, u2, s, right, d2);
            bra += at(t, e1, left, s, d1) * at(t, e2, s, right, d2);
          }
          sum += ket * bra;
        }
  return sum;
}

// T'[u, k1, k2, d] = sum U[l1, l2, k1] Tu[u, l1, r1, s] Td[s, l2, r2, d] U[r1, r2, k2]
double update_y_entry(tensor4 const& upper, tensor4 const& lower, tensor3 const& iso, idx up, idx k1, idx k2, idx down)
{
  idx const n = upper.extent(1), ms = upper.extent(3);
  double sum = 0.0;
  for (idx l1 = 0; l1 < n; ++l1)
    for (idx l2 = 0; l2 < n; ++l2)
      for (idx r1 = 0; r1 < n; ++r1)
        for (idx r2 = 0; r2 < n; ++r2)
        {
          double pair = 0.0;
          for (idx s = 0; s < ms; ++s)
            pair += at(upper, up, l1, r1, s) * at(lower, s, l2, r2, down);
          sum += at(iso, l1, l2, k1) * pair * at(iso, r1, r2, k2);
        }
  return sum;
}

// T'[k1, l, r, k2] = sum U[u1, u2, k1] Tl[u1, l, s, d1] Tr[u2, s, r, d2] U[d1, d2, k2]
double update_x_entry(tensor4 const& t, tensor3 const& iso, idx k1, idx left, idx right, idx k2)
{
  idx const m = t.extent(0), n = t.extent(1);
  double sum = 0.0;
  for (idx u1 = 0; u1 < m; ++u1)
    for (idx u2 = 0; u2 < m; ++u2)
      for (idx d1 = 0; d1 < m; ++d1)
        for (idx d2 = 0; d2 < m; ++d2)
        {
          double pair = 0.0;
          for (idx s = 0; s < n; ++s)
            pair += at(t, u1, left, s, d1) * at(t, u2, s, right, d2);
          sum += at(iso, u1, u2, k1) * pair * at(iso, d1, d2, k2);
        }
  return sum;
}

// The step's own claims about its isometry, against a gram evaluated from the definition.
template <class Entry> void check_isometry(char const* name, step_result const& step, idx n, Entry&& gram_entry)
{
  idx const k = step.U.extent(2);
  double orthonormal = 0.0, eigen = 0.0, trace = 0.0, kept = 0.0;
  for (idx a = 0; a < k; ++a)
    for (idx b = 0; b < k; ++b)
    {
      double dot = 0.0;
      for (idx i = 0; i < n; ++i)
        for (idx j = 0; j < n; ++j)
          dot += at(step.U, i, j, a) * at(step.U, i, j, b);
      orthonormal = std::max(orthonormal, std::abs(dot - (a == b ? 1.0 : 0.0)));
    }
  for (idx i = 0; i < n; ++i)
    for (idx j = 0; j < n; ++j)
    {
      trace += gram_entry(i, j, i, j);
      for (idx a = 0; a < k; ++a)
      {
        double applied = 0.0;
        for (idx e1 = 0; e1 < n; ++e1)
          for (idx e2 = 0; e2 < n; ++e2)
            applied += gram_entry(i, j, e1, e2) * at(step.U, e1, e2, a);
        eigen = std::max(eigen, std::abs(applied - at(step.U, i, j, a) * step.S[static_cast<std::size_t>(a)]));
      }
    }
  bool descending = true;
  for (std::size_t a = 0; a < step.S.size(); ++a)
  {
    kept += step.S[a];
    if (a > 0 && step.S[a] > step.S[a - 1]) descending = false;
  }
  double const discarded = 1.0 - kept / trace;
  d::emit(text("{}: kept {} of {}, |U^T U - 1| {:.1e}, |theta U - U S| {:.1e}, discarded {:.6f} (definition {:.6f})",
               name, static_cast<int>(k), static_cast<int>(n * n), orthonormal, eigen, step.discarded, discarded),
          d::stream::out);
  report(orthonormal < 1e-12 && eigen < 1e-11 && descending && std::abs(discarded - step.discarded) < 1e-12,
         std::string(name) + ": U is the leading eigenbasis of the gram evaluated from the definition");
}

void kernel_against_the_definition()
{
  d::emit(p::report_builder("A  the kernel against the definition (random tensor, vertical dimension 2, horizontal 3)"),
          d::stream::out);
  idx const m = 2, n = 3;
  tensor4 const t = random4(m, n, n, m, 12345);

  {
    tensor4 const theta_y = gram(t, axis::y), theta_x = gram(t, axis::x);
    double worst_y = 0.0, worst_x = 0.0;
    for (idx a = 0; a < n; ++a)
      for (idx b = 0; b < n; ++b)
        for (idx c = 0; c < n; ++c)
          for (idx d = 0; d < n; ++d)
            worst_y = std::max(worst_y, std::abs(at(theta_y, a, b, c, d) - gram_y_entry(t, a, b, c, d)));
    for (idx a = 0; a < m; ++a)
      for (idx b = 0; b < m; ++b)
        for (idx c = 0; c < m; ++c)
          for (idx d = 0; d < m; ++d)
            worst_x = std::max(worst_x, std::abs(at(theta_x, a, b, c, d) - gram_x_entry(t, a, b, c, d)));
    d::emit(text("gram: max |library - definition| along y {:.1e}, along x {:.1e}", worst_y, worst_x), d::stream::out);
    report(worst_y < 1e-12 && worst_x < 1e-12, "gram(T, y) and gram(T, x) are the pictures' sums");
  }

  {
    tensor4 const upper = random4(2, n, n, 4, 777), lower = random4(4, n, n, 5, 778);
    tensor3 iso(n, n, 4);
    generator g{779};
    for (auto& value : iso.storage())
      value = g.next();
    tensor4 const out = apply_isometry(upper, lower, iso);
    double worst = 0.0;
    for (idx up = 0; up < 2; ++up)
      for (idx k1 = 0; k1 < 4; ++k1)
        for (idx k2 = 0; k2 < 4; ++k2)
          for (idx down = 0; down < 5; ++down)
            worst = std::max(worst,
                             std::abs(at(out, up, k1, k2, down) - update_y_entry(upper, lower, iso, up, k1, k2, down)));
    d::emit(
        text("update network, operands (2,3,3,4) over (4,3,3,5), a random 3x3x4 U: max |library - definition| {:.1e}",
             worst),
        d::stream::out);
    report(out.extent(0) == 2 && out.extent(1) == 4 && out.extent(2) == 4 && out.extent(3) == 5 && worst < 1e-12,
           "apply_isometry is the update network for unequal leg dimensions");
  }

  for (std::size_t D : {std::size_t{4}, std::size_t{100}})
  {
    step_result const along_y = coarse_grain(t, axis::y, D);
    check_isometry(D == 4 ? "y step, D 4" : "y step, D 100", along_y, n,
                   [&](idx a, idx b, idx c, idx d) { return gram_y_entry(t, a, b, c, d); });
    idx const k = along_y.U.extent(2);
    double worst = 0.0;
    for (idx up = 0; up < m; ++up)
      for (idx k1 = 0; k1 < k; ++k1)
        for (idx k2 = 0; k2 < k; ++k2)
          for (idx down = 0; down < m; ++down)
            worst = std::max(worst, std::abs(at(along_y.T, up, k1, k2, down) * along_y.norm -
                                             update_y_entry(t, t, along_y.U, up, k1, k2, down)));
    report(worst < 1e-11 && std::abs(max_element(along_y.T) - 1.0) < 1e-15 && (D == 4 ? k == 4 : k == n * n),
           "y step: the coarse tensor is the update network over its largest entry; D above the rank is clamped");
  }
  for (std::size_t D : {std::size_t{3}, std::size_t{100}})
  {
    step_result const along_x = coarse_grain(t, axis::x, D);
    check_isometry(D == 3 ? "x step, D 3" : "x step, D 100", along_x, m,
                   [&](idx a, idx b, idx c, idx d) { return gram_x_entry(t, a, b, c, d); });
    idx const k = along_x.U.extent(2);
    double worst = 0.0;
    for (idx k1 = 0; k1 < k; ++k1)
      for (idx left = 0; left < n; ++left)
        for (idx right = 0; right < n; ++right)
          for (idx k2 = 0; k2 < k; ++k2)
            worst = std::max(worst, std::abs(at(along_x.T, k1, left, right, k2) * along_x.norm -
                                             update_x_entry(t, along_x.U, k1, left, right, k2)));
    report(worst < 1e-11 && along_x.T.extent(1) == n && (D == 3 ? k == 3 : k == m * m),
           "x step: the same step with the axes exchanged, horizontal legs untouched");
  }
}

// ----------------------------------------------------------------------------------------------
// B, C, D  partition functions
// ----------------------------------------------------------------------------------------------

// ln Z of the L x L torus by summing over every spin configuration.
double enumerated_lnz(model which, int q, double K, int L)
{
  int const sites = L * L;
  std::vector<double> weight(static_cast<std::size_t>(q));
  for (int delta = 0; delta < q; ++delta)
    weight[static_cast<std::size_t>(delta)] = which == model::potts
                                                  ? std::exp(delta == 0 ? K : 0.0)
                                                  : std::exp(K * std::cos(2.0 * std::numbers::pi * delta / q));
  std::vector<int> spin(static_cast<std::size_t>(sites), 0);
  double z = 0.0;
  for (;;)
  {
    double w = 1.0;
    for (int y = 0; y < L; ++y)
      for (int x = 0; x < L; ++x)
      {
        int const here = spin[static_cast<std::size_t>(y * L + x)];
        int const right = spin[static_cast<std::size_t>(y * L + (x + 1) % L)];
        int const down = spin[static_cast<std::size_t>(((y + 1) % L) * L + x)];
        w *= weight[static_cast<std::size_t>(((here - right) % q + q) % q)] *
             weight[static_cast<std::size_t>(((here - down) % q + q) % q)];
      }
    z += w;
    int site = 0;
    while (site < sites && ++spin[static_cast<std::size_t>(site)] == q)
      spin[static_cast<std::size_t>(site++)] = 0;
    if (site == sites) break;
  }
  return std::log(z);
}

void exact_tori()
{
  d::emit(p::report_builder("B  untruncated flows against enumerated partition functions"), d::stream::out);
  struct row
  {
      model which;
      char const* name;
      int q;
      double K;
      int steps;
      std::size_t D;
  };
  auto results = p::make_data_table("Enumerated torus comparison", p::data_column<std::string>("model").label("Model"),
                                    p::data_column<double>("coupling").label("K").fixed(10),
                                    p::data_column<int>("width").label("Width"),
                                    p::data_column<double>("error").label("Error in ln Z").scientific(1));
  results.attach(p::terminal_sink());
  for (row const& r :
       {row{model::clock, "Ising", 2, 0.30, 2, 16}, row{model::clock, "Ising", 2, 0.4406867935, 2, 16},
        row{model::clock, "3-state clock", 3, 0.55, 1, 9}, row{model::potts, "3-state Potts", 3, 0.9, 1, 9},
        row{model::clock, "4-state clock", 4, 0.7, 1, 16}})
  {
    flow_record const record = flow(vertex(r.which, r.q, r.K), r.D, r.steps);
    double worst = std::abs(record.unit_lnz - enumerated_lnz(r.which, r.q, r.K, 1));
    results.append(r.name, r.K, 1, worst);
    for (int step = 0; step < r.steps; ++step)
    {
      int const L = 2 << step;
      double const exact = enumerated_lnz(r.which, r.q, r.K, L);
      double const gap = std::abs(record.lnz[static_cast<std::size_t>(step)] * L * L - exact) / exact;
      results.append(r.name, r.K, L, gap);
      worst = std::max(worst, gap);
    }
    report(worst < 1e-12, std::string(r.name) + ": ln Z of the torus equals the sum over spin configurations");
  }
  results.finish();
}

void clock_is_potts_at_q3()
{
  d::emit(p::report_builder("C  the q = 3 clock model at K is the 3-state Potts model at 3K/2"), d::stream::out);
  double const K = 0.6;
  flow_record const clock = flow(vertex(model::clock, 3, K), 9, 8),
                    potts = flow(vertex(model::potts, 3, 1.5 * K), 9, 8);
  double const gap = std::abs(clock.lnz.back() - (potts.lnz.back() - K));
  d::emit(text("D 9, L 256: ln Z / N clock {:.14f}, Potts - K {:.14f}, difference {:.1e}", clock.lnz.back(),
               potts.lnz.back() - K, gap),
          d::stream::out);
  report(gap < 1e-12, "the two models' free energies differ by exactly K per site");
}

void onsager()
{
  d::emit(p::report_builder("D  Onsager's free energy (bond 8, L = 4096)"), d::stream::out);
  // What is left is the truncation: about 3e-6 away from the critical point and 1.4e-5 at it.
  for (auto const& [K, bound] :
       {std::pair{0.40, 1e-5},
        std::pair{critical_coupling<double>(model::clock, 2, uni20::native_precision<double>()), 5e-5}})
  {
    flow_record const record = flow(vertex(model::clock, 2, K), 8, 12);
    double const exact = onsager_lnz(K), error = std::abs(record.lnz.back() - exact) / exact;
    d::emit(text("K {:.10f}: ln Z / N {:.12f}, Onsager {:.12f}, relative error {:.1e} (bound {:.0e})", K,
                 record.lnz.back(), exact, error, bound),
            d::stream::out);
    report(error < bound, "the free energy is Onsager's to the accuracy of a bond of 8");
  }
}

// ----------------------------------------------------------------------------------------------
// E  dispatch
// ----------------------------------------------------------------------------------------------

void every_contraction_in_gemm()
{
  d::emit(p::report_builder("E  where the contractions of one round run"), d::stream::out);
  flow_record const plateau = flow(
      vertex(model::clock, 2, critical_coupling<double>(model::clock, 2, uni20::native_precision<double>())), 12, 6);
  std::vector<std::pair<std::string, std::string>> walks;
  {
    uni20::linalg::dispatch_diagnostics::scoped_sink const watch(
        [&walks](uni20::linalg::dispatch_diagnostics::event const& e) {
          auto const selected = e.selected_backend();
          walks.emplace_back(e.operation, selected ? std::string(*selected) : std::string("NONE"));
        });
    static_cast<void>(one_round(plateau.T0, 12));
  }
  std::vector<std::pair<std::pair<std::string, std::string>, int>> tally;
  bool reference_loop = false;
  int contractions = 0;
  for (auto const& walk : walks)
  {
    auto found = std::find_if(tally.begin(), tally.end(), [&](auto const& entry) { return entry.first == walk; });
    if (found == tally.end())
      tally.emplace_back(walk, 1);
    else
      ++found->second;
    if (walk.first == "contract")
    {
      ++contractions;
      if (walk.second.find("gemm") == std::string::npos) reference_loop = true;
    }
  }
  auto dispatch = d::table("Contraction backends");
  dispatch.column("Operation").column("Backend").column("Calls");
  for (auto const& [walk, count] : tally)
    dispatch.row(walk.first, walk.second, count);
  report(contractions == 12 && !reference_loop, "all 12 contractions of a round reach GEMM, none the reference loop");
}

// ----------------------------------------------------------------------------------------------
// F, G  the transfer matrix against the exactly solved strip
// ----------------------------------------------------------------------------------------------

// After k rounds a coarse tensor spans 2^k rows and 2^k columns, so joining its two horizontal
// legs gives V^(2^k) for a periodic strip of 2^k sites.  Run with no truncation, its scaling
// dimensions must be the exactly solved strip's.
void transfer_matrix_against_the_exact_strip()
{
  d::emit(p::report_builder("F  the coarse tensor's transfer matrix against the exact Ising strip"), d::stream::out);
  for (int rounds : {1, 2})
    for (double over_tc : {1.0, 1.2})
    {
      long long const sites = 1LL << rounds;
      auto const* expected = exact_strip(sites, over_tc);
      if (!expected)
      {
        report(false, "a tabulated strip exists for this width and temperature");
        continue;
      }
      tensor4 t = vertex(model::clock, 2,
                         critical_coupling<double>(model::clock, 2, uni20::native_precision<double>()) / over_tc)
                      .tensor;
      for (int step = 0; step < rounds; ++step)
        t = one_round(t, no_truncation);

      auto const spectrum = ring_levels(t);
      auto const X = scaling_dimensions(spectrum.eigenvalues);
      std::size_t const compared = std::min(X.size(), expected->scaling_dimensions.size());
      double absolute = 0.0;
      double const relative = worst_relative_to_resolution(X, expected->scaling_dimensions, compared, &absolute);
      d::emit(text("{} round{} -> {} sites at {:.1f} Tc: bond {}, all {} levels compared, max |X - exact| {:.1e} = "
                   "{:.2f} of what is resolvable",
                   rounds, rounds == 1 ? " " : "s", sites, over_tc, static_cast<int>(t.extent(0)), compared, absolute,
                   relative),
              d::stream::out);
      report(spectrum.asymmetry < 1e-12, "the transfer matrix of a reflection-symmetric tensor is symmetric");
      report(compared == expected->scaling_dimensions.size() && relative < 1.0,
             "its scaling dimensions are the exactly solved strip's");
    }
}

// This is how the method is used: keep the bond fixed and read the strip at every round.  While
// the bond is wide enough the reading is still exact; past that the truncation shows.  At T_c
// a bond of 8 keeps the numbers near the Ising CFT values X_sigma = 1/8 and X_eps = 1 up to
// about 16 sites, after which X_eps drifts away; the default run shows a bond of 16 holding
// them further.  Only the lowest few levels are compared: a bond of D carries D levels, against
// the strip's 2^n.
constexpr std::size_t lowest_levels = 4;

void truncated_flow_against_the_exact_strip()
{
  d::emit(p::report_builder(
              fmt::format("G  a truncated flow read width by width (bond 8, the lowest {} levels)", lowest_levels)),
          d::stream::out);
  std::size_t const D = 8;
  for (double over_tc : {1.0, 1.2})
  {
    auto results = p::make_data_table(fmt::format("Strip spectrum at T/Tc = {:.1f}", over_tc),
                                      p::data_column<long long>("width").label("Width"),
                                      p::data_column<double>("x1").label("X_1").fixed(6),
                                      p::data_column<double>("exact_x1").label("Exact X_1").fixed(6),
                                      p::data_column<double>("x2").label("X_2").fixed(6),
                                      p::data_column<double>("exact_x2").label("Exact X_2").fixed(6),
                                      p::data_column<double>("error").label("Max error").scientific(1));
    results.attach(p::terminal_sink());
    tensor4 t =
        vertex(model::clock, 2, critical_coupling<double>(model::clock, 2, uni20::native_precision<double>()) / over_tc)
            .tensor;
    double exact_while_wide = 0.0, last_first = 0.0, first_at_16 = 0.0, second_at_16 = 0.0;
    long long last_sites = 0;
    for (int rounds = 1; rounds <= 6; ++rounds)
    {
      bool const still_exact = static_cast<std::size_t>(t.extent(0)) * static_cast<std::size_t>(t.extent(0)) <= D;
      t = one_round(t, D);
      long long const sites = 1LL << rounds;
      auto const X = scaling_dimensions(ring_levels(t).eigenvalues);
      auto const* expected = exact_strip(sites, over_tc);
      if (X.size() < lowest_levels || !expected) continue;
      double absolute = 0.0;
      worst_relative_to_resolution(X, expected->scaling_dimensions, lowest_levels, &absolute);
      results.append(sites, X[1], uni20::parse_real<double>(expected->scaling_dimensions[1]), X[2],
                     uni20::parse_real<double>(expected->scaling_dimensions[2]), absolute);
      if (still_exact) exact_while_wide = std::max(exact_while_wide, absolute);
      if (sites == 16)
      {
        first_at_16 = X[1];
        second_at_16 = X[2];
      }
      last_first = X[1];
      last_sites = sites;
    }
    results.finish();
    report(exact_while_wide < 1e-8, "while the bond is wide enough the reading is still the exact strip's");
    if (over_tc == 1.0)
    {
      d::emit(text("The critical Ising CFT has X_sigma = 1/8 and X_eps = 1."), d::stream::out);
      report(std::abs(first_at_16 - 0.125) < 0.005 && std::abs(second_at_16 - 1.0) < 0.08,
             "at T_c the reading at 16 sites is near the Ising CFT dimensions");
    }
    else
    {
      d::emit(text("Above T_c the strip is disordered, so X grows with the width."), d::stream::out);
      report(last_first > 1.0 && last_sites >= 32, "above T_c the reading leaves the critical values");
    }
  }
}

// ----------------------------------------------------------------------------------------------
// H  inputs at the edge
// ----------------------------------------------------------------------------------------------

void inputs_at_the_edge()
{
  d::emit(p::report_builder("H  inputs at the edge"), d::stream::out);

  // A tensor that is zero unless both horizontal legs are in their first state: the gram of
  // the fused pair has rank one, so a cut that keeps two states falls inside its null space.
  {
    idx const m = 2, n = 3;
    tensor4 t(m, n, n, m);
    generator g{2468};
    for (idx up = 0; up < m; ++up)
      for (idx down = 0; down < m; ++down)
        t[up, 0, 0, down] = std::abs(g.next()) + 0.1;
    step_result const step = coarse_grain(t, axis::y, 2);
    d::emit(text("rank-one gram, 2 states kept: singular values {:.3e} and {:.3e}, gap {}", step.S.front(),
                 step.S.back(), step.gap),
            d::stream::out);
    report(step.S.size() == 2 && std::isfinite(step.gap) && step.gap >= 0.0 && step.gap <= 1.0,
           "a cut inside the null space of the gram reports a finite gap");
  }

  auto const unit = vertex(model::clock, 2, 0.4);
  auto const record = flow(unit, 1, 128);
  report(record.log2_width.back() == 128 && std::isfinite(record.lnz.back()),
         "128 rounds retain the lattice width as an exponent without overflow");
}

// The largest difference relative to the largest entry of `a`; infinite if any entry of either is not
// finite, so that an unwritten or corrupted entry cannot pass.
template <class A, class B> double relative_difference(A const& a, B const& b)
{
  double worst = 0.0, scale = 0.0;
  auto other = b.storage().begin();
  for (auto const& value : a.storage())
  {
    double const x = static_cast<double>(value), y = static_cast<double>(*other);
    if (!std::isfinite(x) || !std::isfinite(y)) return HUGE_VAL;
    scale = std::max(scale, std::abs(x));
    worst = std::max(worst, std::abs(x - y));
    ++other;
  }
  return worst / scale;
}

// The largest difference between two lists; infinite if their lengths differ or an entry is not finite.
double largest_difference(std::vector<double> const& a, std::vector<double> const& b)
{
  if (a.size() != b.size()) return HUGE_VAL;
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return HUGE_VAL;
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
// ----------------------------------------------------------------------------------------------
// I  the step on CUDA tensors
// ----------------------------------------------------------------------------------------------

// The device named by --second-cuda-device, enrolled beside the selected one for the placement check.
std::optional<int> second_cuda_device;

void the_step_on_cuda_tensors()
{
  d::emit(p::report_builder("I  the step on CUDA tensors against the host"), d::stream::out);
  {
    tensor4 const t = random4(3, 4, 4, 3, 4321);
    double const off = relative_difference(gram_y(t), uni20::to_host(gram_y(device_place{}.upload(t))));
    d::emit(text("gram of a (3, 4, 4, 3) tensor: off the host by {:.2e} of its largest entry", off), d::stream::out);
    report(off < 1e-13, "the gram of a CUDA tensor is the host's");
  }
  {
    tensor4 const upper = random4(3, 4, 4, 5, 4322), lower = random4(5, 4, 4, 2, 4323);
    tensor3 iso(4, 4, 6);
    generator g{4324};
    for (auto& value : iso.storage())
      value = g.next();
    auto const on_device =
        apply_isometry(device_place{}.upload(upper), device_place{}.upload(lower), device_place{}.upload(iso));
    double const off = relative_difference(apply_isometry(upper, lower, iso), uni20::to_host(on_device));
    d::emit(text("update network, upper (3, 4, 4, 5), lower (5, 4, 4, 2), isometry (4, 4, 6): off by {:.2e}", off),
            d::stream::out);
    report(off < 1e-13, "the update network on CUDA tensors is the host's");
  }
  double const K = critical_coupling<double>(model::clock, 2, uni20::native_precision<double>());
  {
    flow_record const start = flow(vertex(model::clock, 2, K), 4, 2); // a generic 4^4 tensor
    auto const t_device = device_place{}.upload(start.T0);
    for (axis along : {axis::y, axis::x})
    {
      char const* const name = along == axis::y ? "y" : "x";
      step_result const expected = coarse_grain(start.T0, along, 6);
      auto const got = coarse_grain(t_device, along, 6);
      double const invariant = expected.norm * torus_trace(expected.T);
      double const spectrum = largest_difference(expected.S, got.S) / expected.S.front();
      double const weight = std::abs(expected.discarded - got.discarded);
      double const gap = std::abs(expected.gap - got.gap);
      double const trace = std::abs(got.norm * torus_trace(got.T) - invariant) / std::abs(invariant);
      d::emit(text("{} step: spectrum {:.1e}, discarded weight {:.1e}, gap {:.1e}, norm x trace {:.1e}", name, spectrum,
                   weight, gap, trace),
              d::stream::out);
      report(spectrum < 1e-12 && weight < 1e-12 && gap < 1e-9 && trace < 1e-12,
             std::string("the ") + name + " step on CUDA tensors agrees with the host");
    }
  }
  for (std::size_t D : {std::size_t{3}, std::size_t{8}})
  {
    int const rounds = 6;
    flow_record const expected = flow(vertex(model::clock, 2, K), D, rounds);
    flow_record const got = flow<device_place>(vertex(model::clock, 2, K), D, rounds);
    double const lnz = largest_difference(expected.lnz, got.lnz);
    double const cut = std::max(largest_difference(expected.discarded_y, got.discarded_y),
                                largest_difference(expected.discarded_x, got.discarded_x));
    d::emit(text("bond {}, {} rounds: ln Z / N {:.15f}, off the host by {:.1e}; cut errors off by {:.1e}", D, rounds,
                 got.lnz.back(), lnz, cut),
            d::stream::out);
    report(lnz < 1e-11 && cut < 1e-9, "bond " + std::to_string(D) + ": the flow on CUDA tensors is the host's");
  }
  {
    // The eigh cut on the device (cuSOLVER syevd) keeps the subspace of the host's SVD cut.
    int const rounds = 6;
    flow_record const expected = flow(vertex(model::clock, 2, K), 8, rounds);
    flow_record const got = flow<device_place>(vertex(model::clock, 2, K), 8, rounds, nullptr, cut_method::eigh);
    double const lnz = largest_difference(expected.lnz, got.lnz);
    double const cut = std::max(largest_difference(expected.discarded_y, got.discarded_y),
                                largest_difference(expected.discarded_x, got.discarded_x));
    d::emit(text("bond 8, eigh cut: ln Z / N off the host's SVD cut by {:.1e}; cut errors off by {:.1e}", lnz, cut),
            d::stream::out);
    report(lnz < 1e-11 && cut < 1e-9, "the eigh cut on CUDA tensors keeps the subspace of the host's SVD cut");
    flow_record const selected =
        flow<device_place>(vertex(model::clock, 2, K), 8, rounds, nullptr, cut_method::selected_eigh);
    double const selected_lnz = largest_difference(expected.lnz, selected.lnz);
    d::emit(text("bond 8, selected-eigh cut: ln Z / N off the host's SVD cut by {:.1e}", selected_lnz), d::stream::out);
    report(selected_lnz < 1e-11, "the selected-eigh cut on CUDA tensors keeps the subspace of the host's SVD cut");
    for (cut_method method : {cut_method::jacobi_eigh, cut_method::jacobi_svd})
    {
      flow_record const jacobi = flow<device_place>(vertex(model::clock, 2, K), 8, rounds, nullptr, method);
      double const jacobi_lnz = largest_difference(expected.lnz, jacobi.lnz);
      d::emit(text("bond 8, {} cut: ln Z / N off the host's SVD cut by {:.1e}", cut_method_name(method), jacobi_lnz),
              d::stream::out);
      report(jacobi_lnz < 1e-11, std::string("the ") + cut_method_name(method) +
                                     " cut on CUDA tensors keeps the subspace of the host's SVD cut");
    }
  }
  if (second_cuda_device)
  {
    // A step on a tensor held by a device that is not the runtime's default stays on that device.
    int const other = *second_cuda_device;
    flow_record const start = flow(vertex(model::clock, 2, K), 4, 2);
    auto const t_other = uni20::to_device(start.T0, other);
    step_result const expected = coarse_grain(start.T0, axis::x, 6);
    auto const got = coarse_grain(t_other, axis::x, 6);
    bool const placed = place_of(got.T).resources->device().ordinal() == other &&
                        place_of(got.U).resources->device().ordinal() == other;
    double const spectrum = largest_difference(expected.S, got.S) / expected.S.front();
    d::emit(text("x step on device {}: results on devices {} and {}, spectrum off the host by {:.1e}", other,
                 place_of(got.T).resources->device().ordinal(), place_of(got.U).resources->device().ordinal(),
                 spectrum),
            d::stream::out);
    report(placed && spectrum < 1e-12, "a step on another device keeps its results there");
  }
  {
    // The observer of a CUDA flow sees the same record as on the host: T0 is the tensor it is given.
    bool consistent = true;
    int observed = 0;
    static_cast<void>(flow<device_place>(vertex(model::clock, 2, K), 4, 3, [&](auto const& r, auto const& t) {
      consistent = consistent && &r.T0 == &t;
      ++observed;
    }));
    report(consistent && observed == 3, "the observer of a CUDA flow reads the latest tensor in record.T0");
  }
}
#endif

int run_checks(bool on_device)
{
  d::emit(p::report_builder("Dense HOTRG checks"), d::stream::out);
  kernel_against_the_definition();
  exact_tori();
  clock_is_potts_at_q3();
  onsager();
  every_contraction_in_gemm();
  transfer_matrix_against_the_exact_strip();
  truncated_flow_against_the_exact_strip();
  inputs_at_the_edge();
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
  if (on_device) the_step_on_cuda_tensors();
#else
  static_cast<void>(on_device);
#endif
  p::report_builder summary("HOTRG checks");
  summary
      .status(failures == 0 ? p::semantic_glyph::success : p::semantic_glyph::failure,
              failures == 0 ? "All checks passed" : "Some checks failed")
      .field("Failed", failures);
  d::emit(std::move(summary), d::stream::out);
  return failures == 0 ? 0 : 1;
}

// ----------------------------------------------------------------------------------------------
// arguments
// ----------------------------------------------------------------------------------------------

template <uni20::Real R> R read_real(std::string_view text, uni20::Precision p)
{
  if constexpr (requires { uni20::parse_real<R>(text, p); })
    return uni20::parse_real<R>(text, p);
  else
    return uni20::parse_real<R>(text);
}

void report_sketch(p::report_builder& setup, sketch_options const& sketch)
{
  setup.field("Seed", sketch.seed)
      .field("Oversampling", sketch.oversampling)
      .field("Power iterations", sketch.power_iterations);
}

template <uni20::Real R, class Place = host_place>
int run_flow(std::size_t D, int rounds, std::string_view temperature, uni20::Precision precision, int quadrature_points,
             cut_method method, sketch_options const& sketch)
{
  R const ratio = read_real<R>(temperature, precision);
  if (!uni20::isfinite(ratio) || ratio <= 0) throw std::invalid_argument("temperature must be finite and positive");
  R const K = critical_coupling<R>(model::clock, 2, precision) / ratio;
  R const exact = onsager_lnz(K, quadrature_points);
  p::report_builder setup("Dense HOTRG, 2D Ising");
  setup.field("T/Tc", uni20::format_scalar(ratio))
      .field("Bits", precision.bit_count())
      .field("Bond", D)
      .field("Rounds", rounds)
      .field("Width", "2^log2(width)")
      .field("Tensors", Place::name)
      .field("Gram cut", cut_method_name(method));
  if (uses_sketch(method)) report_sketch(setup, sketch);
  d::emit(std::move(setup), d::stream::out);
  auto results = make_flow_table<R>(precision);
  results.attach(p::terminal_sink());
  auto const record = flow<Place>(
      vertex(model::clock, 2, K), D, rounds,
      [&](auto const& r, auto const& t) {
        auto const i = r.log2_width.size() - 1;
        int const exponent = r.log2_width[i];
        auto const spectrum = ring_levels(t);
        auto const X = scaling_dimensions(spectrum.eigenvalues);
        auto const* expected = reference::find(exponent, ratio);
        auto level = [&](std::size_t n) -> std::optional<R> {
          if (n < X.size()) return X[n];
          return std::nullopt;
        };
        auto reference_level = [&](std::size_t n) -> std::optional<R> {
          if (expected) return displayed_reference<R>(expected->scaling_dimensions[n], precision);
          return std::nullopt;
        };
        results.append(exponent, t.extent(0), std::max(r.discarded_y[i], r.discarded_x[i]), r.lnz[i],
                       R((r.lnz[i] - exact) / exact), level(1), reference_level(1), level(2), reference_level(2),
                       spectrum.asymmetry);
      },
      method, sketch);
  results.finish();
  double all = 0, contractions = 0, regroup = 0, svd = 0;
  for (std::size_t i = 0; i < record.seconds.size(); ++i)
  {
    all += record.seconds[i];
    contractions += record.seconds_contract[i];
    regroup += record.seconds_regroup[i];
    svd += record.seconds_cut[i];
  }
  auto timing = p::make_data_table("Timing", p::data_column<std::string>("phase").label("Phase"),
                                   p::data_column<double>("seconds").label("Seconds").fixed(2));
  timing.append("Total", all);
  timing.append("Contractions", contractions);
  timing.append("Axis regroupings", regroup);
  timing.append("Gram cut", svd);
  timing.finish();
  p::report_builder timing_report;
  timing_report.table("") = p::to_report_table(timing);
  d::emit(std::move(timing_report), d::stream::out);
  p::report_builder diagnostics("Reference diagnostics");
  diagnostics.field("vs Onsager", "Finite torus compared with the infinite lattice")
      .field("Onsager midpoint samples", quadrature_points)
      .field("Error sources", "Quadrature and bond truncation errors are separate from arithmetic precision")
      .field("Strip references", "100 significant digits");
  d::emit(std::move(diagnostics), d::stream::out);
  return record.log2_width.size() == static_cast<std::size_t>(rounds) ? 0 : 1;
}

// ----------------------------------------------------------------------------------------------
// --benchmark-gram-cut: every cut method on the same grams, and a flow with each
// ----------------------------------------------------------------------------------------------

// ||U U^T - V V^T||_F for isometries U and V given as [n, n, k] host tensors: zero for the same kept
// subspace, at most sqrt(2 k).  It is sqrt(2) ||U - V (V^T U)||_F, which, unlike 2 k - 2 ||V^T U||_F^2,
// does not cancel to sqrt(epsilon) for nearly equal subspaces.  The subspace is the model's only where
// the gap at the cut is not tiny.
double subspace_distance(tensor3 const& U, tensor3 const& V)
{
  idx const n = U.extent(0), k = U.extent(2);
  std::vector<double> overlap(static_cast<std::size_t>(k * k), 0.0); // (V^T U)[e, c]
  for (idx c = 0; c < k; ++c)
    for (idx e = 0; e < k; ++e)
      for (idx b = 0; b < n; ++b)
        for (idx a = 0; a < n; ++a)
          overlap[static_cast<std::size_t>(e + k * c)] += at(V, a, b, e) * at(U, a, b, c);
  double residual = 0.0;
  for (idx c = 0; c < k; ++c)
    for (idx b = 0; b < n; ++b)
      for (idx a = 0; a < n; ++a)
      {
        double r = at(U, a, b, c);
        for (idx e = 0; e < k; ++e)
          r -= at(V, a, b, e) * overlap[static_cast<std::size_t>(e + k * c)];
        residual += r * r;
      }
  return std::sqrt(2.0 * residual);
}

template <class Place>
int run_cut_benchmark(std::vector<std::size_t> const& bonds, int rounds, int repeat, sketch_options const& sketch)
{
  // Every method with an implementation for these tensors: the Jacobi solvers are CUDA only, and the
  // divide-and-conquer drivers host only.
  std::vector<cut_method> methods;
  for (cut_method method : all_cut_methods)
    if (std::same_as<Place, host_place> ? !cut_method_needs_device(method) : !cut_method_needs_host(method))
      methods.push_back(method);
  std::size_t method_width = 0;
  for (cut_method method : methods)
    method_width = std::max(method_width, std::string_view(cut_method_name(method)).size());
  double const K = critical_coupling<double>(model::clock, 2, uni20::native_precision<double>());
  using clock = std::chrono::steady_clock;

  p::report_builder setup("Gram-cut benchmark, critical 2D Ising, fp64");
  setup.field("Tensors", Place::name)
      .field("Gram", "the y-step gram of a 4-round flow's coarse tensor, (D^2 x D^2)")
      .field("Reference", "host eigh of the same gram")
      .field("Repeats", repeat);
  report_sketch(setup, sketch);
  d::emit(std::move(setup), d::stream::out);

  auto cuts = p::make_data_table(
      "Cut of one gram", {.retain = p::retention::none}, p::data_column<idx>("bond").label("Bond"),
      p::data_column<std::string>("method").label("Method").width(uni20::display::width::fit(method_width)),
      p::data_column<double>("seconds").label("Seconds").scientific(2),
      p::data_column<double>("spectrum").label("Spectrum error").scientific(2),
      p::data_column<double>("discarded").label("Discarded error").scientific(2),
      p::data_column<double>("subspace").label("Subspace distance").scientific(2),
      p::data_column<double>("gap").label("Gap").scientific(2));
  cuts.attach(p::terminal_sink());
  for (std::size_t D : bonds)
  {
    flow_record const plateau = flow(vertex(model::clock, 2, K), D, 4);
    tensor4 const theta = gram(plateau.T0, axis::y);
    auto const reference = detail::cut_gram(theta, D, cut_method::eigh);
    for (cut_method method : methods)
    {
      double best = HUGE_VAL;
      std::optional<tensor3> U;
      std::vector<double> S;
      double discarded = 0.0, gap = 0.0;
      for (int r = 0; r < repeat; ++r)
      {
        if constexpr (std::same_as<Place, host_place>)
        {
          auto const start = clock::now();
          auto cut = detail::cut_gram(theta, D, method, sketch, {.round = 4, .along = axis::y});
          best = std::min(best, std::chrono::duration<double>(clock::now() - start).count());
          U.emplace(std::move(cut.U));
          S = std::move(cut.S);
          discarded = cut.discarded;
          gap = cut.gap;
        }
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
        else
        {
          auto on_device = Place{}.upload(theta);
          wait_for(on_device);
          auto const start = clock::now();
          auto cut = detail::cut_gram(std::move(on_device), D, method, sketch, {.round = 4, .along = axis::y});
          wait_for(cut.U);
          best = std::min(best, std::chrono::duration<double>(clock::now() - start).count());
          U.emplace(uni20::to_host(cut.U));
          S = std::move(cut.S);
          discarded = cut.discarded;
          gap = cut.gap;
        }
#endif
      }
      cuts.append(static_cast<idx>(D), std::string(cut_method_name(method)), best,
                  largest_difference(reference.S, S) / reference.S.front(), std::abs(discarded - reference.discarded),
                  subspace_distance(*U, reference.U), gap);
    }
  }
  cuts.finish();

  auto flows = p::make_data_table(
      "Flow with each cut", {.retain = p::retention::none}, p::data_column<idx>("bond").label("Bond"),
      p::data_column<std::string>("method").label("Method").width(uni20::display::width::fit(method_width)),
      p::data_column<double>("seconds").label("Seconds").scientific(2), p::data_column<double>("lnz").label("ln Z/N"),
      p::data_column<double>("lnz_vs_svd").label("vs svd").scientific(2), p::data_column<double>("x1").label("X_1"),
      p::data_column<double>("x1_vs_svd").label("vs svd").scientific(2));
  flows.attach(p::terminal_sink());
  for (std::size_t D : bonds)
  {
    double lnz_svd = 0.0, x1_svd = 0.0;
    for (cut_method method : methods)
    {
      auto const start = clock::now();
      flow_record const record = flow<Place>(vertex(model::clock, 2, K), D, rounds, nullptr, method, sketch);
      double const seconds = std::chrono::duration<double>(clock::now() - start).count();
      auto const X = scaling_dimensions(ring_levels(record.T0).eigenvalues);
      double const lnz = record.lnz.back(), x1 = X.size() > 1 ? X[1] : HUGE_VAL;
      if (method == cut_method::svd)
      {
        lnz_svd = lnz;
        x1_svd = x1;
      }
      flows.append(static_cast<idx>(D), std::string(cut_method_name(method)), seconds, lnz, std::abs(lnz - lnz_svd), x1,
                   std::abs(x1 - x1_svd));
    }
  }
  flows.finish();
  return 0;
}

[[noreturn]] void usage(char const* complaint)
{
  if (complaint) d::failure("ising_hotrg_example: {}", complaint);
  p::report_builder help("ising_hotrg_example");
  help.field("Usage", "ising_hotrg_example [options]");
  help.table("Options")
      .preserve_tokens()
      .column("Option", p::table_alignment::left)
      .column("Meaning", p::table_alignment::left)
      .row("--bond=D", "Bond dimension (default 16)")
      .row("--rounds=N", "Rounds; represented width is 2^N (default 8)")
      .row("--temperature=X", "Decimal T/Tc, rounded at the selected precision (default 1)")
      .row("--precision=P", "fp32|fp64|fp80|fp128|mpreal (default fp64); wide types require configured MPLAPACK")
      .row("--bits=N", "Binary working precision for mpreal (default 128)")
      .row("--onsager-points=N", "Midpoint samples for the free-energy diagnostic (default 400000)")
      .row("--execution=E", "cpu|cuda (default cpu); cuda takes fp32 or fp64 and needs the CUDA backends")
      .row("--cuda-device=N", "Device ordinal for --execution=cuda (default 0)")
      .row("--second-cuda-device=N", "With --execution=cuda --check: also check a step on a tensor on device N")
      .row("--check", "Original double-precision definition and flow checks; with cuda, also the CUDA step")
      .row("--gram-cut=M",
           "svd|eigh|selected-eigh|jacobi-eigh|jacobi-svd|dc-eigh|dc-svd|rsvd|nystrom (default svd): how a step "
           "takes the leading vectors of the gram; the jacobi methods need --execution=cuda, and the "
           "dc methods --execution=cpu")
      .row("--seed=N", "Seed for randomized cuts, including the benchmark (default 1)")
      .row("--oversampling=P", "Extra Gaussian columns for rsvd and nystrom (default 8)")
      .row("--power-iterations=Q", "Stabilized power steps for rsvd and nystrom (default 3)")
      .row("--benchmark-gram-cut", "Time every gram-cut method on the same grams, and a flow with each (fp64)")
      .row("--bonds=D,D,...", "Bonds for --benchmark-gram-cut (default 8,16,24)")
      .row("--repeat=N", "Timed repeats per cut for --benchmark-gram-cut; the fastest is shown (default 3)")
      .row("-h, --help", "Show this help");
  d::emit(std::move(help), complaint ? d::stream::err : d::stream::out);
  std::exit(complaint ? 2 : 0);
}

template <class Integer> Integer parse_integer(std::string_view text, Integer least = 1)
{
  Integer value{};
  auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value < least)
    throw std::invalid_argument(least == 0 ? "expected a nonnegative integer" : "expected a positive integer");
  return value;
}

#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
// The exit status when the requested CUDA device does not exist; CTest reads it as a skip.
constexpr int no_cuda_device = 77;

// Whether a context can be made on a visible device: a device in exclusive-process mode that
// another process holds, for one, is visible but not usable.  The device is left current, so that
// no other device's context is opened by switching back to it.
bool cuda_device_usable(int ordinal)
{
  bool const usable = cudaSetDevice(ordinal) == cudaSuccess && cudaFree(nullptr) == cudaSuccess;
  static_cast<void>(cudaGetLastError());
  return usable;
}

// Whether `device` is visible and usable; if not, say so.
bool cuda_device_available(int device)
{
  int count = 0;
  if (cudaError_t const status = cudaGetDeviceCount(&count); status != cudaSuccess || device >= count)
  {
    d::failure("ising_hotrg_example: CUDA device {} is not available ({})", device,
               status != cudaSuccess ? std::string(cudaGetErrorString(status)) : std::to_string(count) + " devices");
    return false;
  }
  if (!cuda_device_usable(device))
  {
    d::failure("ising_hotrg_example: CUDA device {} is visible but not usable", device);
    return false;
  }
  return true;
}

// Initialize the CUDA runtime on `device` for the lifetime of `run`, enrolling `second` as well when
// it is given (--second-cuda-device, for the placement check).
template <class Run> int on_cuda_device(int device, std::optional<int> second, Run&& run)
{
  if (!cuda_device_available(device)) return no_cuda_device;
  if (second)
  {
    bool const available = cuda_device_available(*second);
    static_cast<void>(cudaSetDevice(device)); // its context exists already
    if (!available) return no_cuda_device;
  }
  std::vector<int> ordinals{device};
  if (second) ordinals.push_back(*second);
  second_cuda_device = second;
  auto runtime = uni20::cuda::initialize(
      {.device_ordinals = std::move(ordinals), .default_device = device, .streams_per_device = 4});
  uni20::cuda::ScopedDevice const selected(device);
  return run();
}
#endif

} // namespace

int main(int argc, char** argv)
{
  std::size_t bond = 16;
  int rounds = 8, points = 400000;
  [[maybe_unused]] std::int64_t bits = 128;
  std::string_view temperature = "1", precision = "fp64", execution = "cpu";
  [[maybe_unused]] int cuda_device = 0;
  [[maybe_unused]] std::optional<int> second_device;
  cut_method method = cut_method::svd;
  sketch_options sketch;
  std::vector<std::size_t> bonds{8, 16, 24};
  int repeat = 3;
  bool check = false, explicit_bits = false, benchmark = false;
  try
  {
    for (int i = 1; i < argc; ++i)
    {
      std::string_view const argument = argv[i];
      if (argument == "--help" || argument == "-h") usage(nullptr);
      if (argument == "--check")
      {
        check = true;
        continue;
      }
      if (argument == "--benchmark-gram-cut")
      {
        benchmark = true;
        continue;
      }
      auto const split = argument.find('=');
      auto const name = argument.substr(0, split);
      if (split == std::string_view::npos) usage("expected --name=value");
      auto const value = argument.substr(split + 1);
      if (name == "--bond")
        bond = parse_integer<std::size_t>(value);
      else if (name == "--rounds")
        rounds = parse_integer<int>(value);
      else if (name == "--temperature")
        temperature = value;
      else if (name == "--precision")
        precision = value;
      else if (name == "--bits")
      {
        bits = parse_integer<std::int64_t>(value);
        explicit_bits = true;
      }
      else if (name == "--onsager-points")
        points = parse_integer<int>(value);
      else if (name == "--execution")
        execution = value;
      else if (name == "--cuda-device")
        cuda_device = parse_integer<int>(value, 0);
      else if (name == "--second-cuda-device")
        second_device = parse_integer<int>(value, 0);
      else if (name == "--seed")
        sketch.seed = parse_integer<std::uint64_t>(value, 0);
      else if (name == "--oversampling")
        sketch.oversampling = parse_integer<idx>(value, 0);
      else if (name == "--power-iterations")
        sketch.power_iterations = parse_integer<idx>(value, 0);
      else if (name == "--gram-cut")
      {
        auto it = std::find_if(all_cut_methods.begin(), all_cut_methods.end(),
                               [&](cut_method m) { return value == cut_method_name(m); });
        if (it == all_cut_methods.end()) usage("unknown --gram-cut method");
        method = *it;
      }
      else if (name == "--bonds")
      {
        bonds.clear();
        for (std::string_view rest = value; !rest.empty();)
        {
          auto const comma = rest.find(',');
          bonds.push_back(parse_integer<std::size_t>(rest.substr(0, comma)));
          rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        }
      }
      else if (name == "--repeat")
        repeat = parse_integer<int>(value);
      else
        usage("unknown option");
    }
    if (explicit_bits && precision != "mpreal") usage("--bits requires --precision=mpreal");
    if (execution != "cpu" && execution != "cuda") usage("--execution is cpu or cuda");
    if (execution == "cpu" && cut_method_needs_device(method))
      usage("--gram-cut=jacobi-eigh and jacobi-svd run only with --execution=cuda");
    if (execution == "cuda" && cut_method_needs_host(method))
      usage("--gram-cut=dc-eigh and dc-svd run only with --execution=cpu");
    if (uses_sketch(method) && precision != "fp32" && precision != "fp64")
      usage("rsvd and nystrom require --precision=fp32 or fp64");
    if (check && precision != "fp64")
      usage("--check runs the original double checks; use uni20_models_tests for precision coverage");
    if (benchmark && (check || precision != "fp64"))
      usage("--benchmark-gram-cut runs at fp64 and cannot be combined with --check");
    if (second_device && (execution != "cuda" || !check || *second_device == cuda_device))
      usage("--second-cuda-device names a device other than --cuda-device for --execution=cuda --check");
    if (execution == "cuda")
    {
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
      if (check) return on_cuda_device(cuda_device, second_device, [] { return run_checks(true); });
      if (benchmark)
        return on_cuda_device(cuda_device, std::nullopt,
                              [&] { return run_cut_benchmark<device_place>(bonds, rounds, repeat, sketch); });
      if (precision == "fp32")
        return on_cuda_device(cuda_device, std::nullopt, [&] {
          return run_flow<float, device_place>(bond, rounds, temperature, uni20::native_precision<float>(), points,
                                               method, sketch);
        });
      if (precision == "fp64")
        return on_cuda_device(cuda_device, std::nullopt, [&] {
          return run_flow<double, device_place>(bond, rounds, temperature, uni20::native_precision<double>(), points,
                                                method, sketch);
        });
      usage("--execution=cuda takes --precision=fp32 or fp64");
#else
      usage("--execution=cuda needs a build with the CUDA, cuBLAS and cuSOLVER backends");
#endif
    }
    if (check) return run_checks(false);
    if (benchmark) return run_cut_benchmark<host_place>(bonds, rounds, repeat, sketch);
    if (precision == "fp32")
      return run_flow<float>(bond, rounds, temperature, uni20::native_precision<float>(), points, method, sketch);
    if (precision == "fp64")
      return run_flow<double>(bond, rounds, temperature, uni20::native_precision<double>(), points, method, sketch);
#if UNI20_HAS_MPLAPACK_FLOAT80
    if (precision == "fp80")
      return run_flow<uni20::float80>(bond, rounds, temperature, uni20::native_precision<uni20::float80>(), points,
                                      method, sketch);
#endif
#if UNI20_HAS_MPLAPACK_FLOAT128
    if (precision == "fp128")
      return run_flow<uni20::float128>(bond, rounds, temperature, uni20::native_precision<uni20::float128>(), points,
                                       method, sketch);
#endif
#if UNI20_HAS_MPLAPACK_MPREAL
    if (precision == "mpreal")
      return run_flow<uni20::mpreal>(bond, rounds, temperature, uni20::Precision::bits(bits), points, method, sketch);
#endif
    usage("unknown precision or its LAPACK provider is not configured");
  }
  catch (std::exception const& error)
  {
    d::failure("ising_hotrg_example: {}", error.what());
    return 1;
  }
}
