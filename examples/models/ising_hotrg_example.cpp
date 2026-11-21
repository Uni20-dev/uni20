// ising_hotrg_example.cpp -- a dense 2D HOTRG of the square-lattice Ising model.
//
// Without arguments it runs one flow at the critical temperature and prints, round by round,
// the free energy against Onsager's and the two lowest scaling dimensions read from the coarse
// tensor's transfer matrix against the exactly solved strip of the same width.
//
//   ising_hotrg_example [--bond=D] [--rounds=N] [--temperature=T/Tc] [--check]
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
#include "dense_hotrg.hpp"
#include "ising_strip_reference.hpp"

#include <uni20/linalg/dispatch_diagnostics.hpp>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>

using namespace uni20::examples::dense_hotrg;

namespace
{

int failures = 0;

void report(bool ok, std::string const& what)
{
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

// ----------------------------------------------------------------------------------------------
// shared by the flow and by the checks
// ----------------------------------------------------------------------------------------------

constexpr std::size_t no_truncation = std::size_t{1} << 20;

reference::strip const* exact_strip(long long sites, double over_tc)
{
  for (auto const& candidate : reference::strips)
    if (candidate.sites == sites && std::abs(candidate.temperature_over_tc - over_tc) < 1e-12) return &candidate;
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
double worst_relative_to_resolution(std::vector<double> const& X, double const* expected, std::size_t count,
                                    double* worst_absolute = nullptr)
{
  double worst = 0.0, absolute = 0.0;
  for (std::size_t i = 0; i < count; ++i)
  {
    double const gap = std::abs(X[i] - expected[i]);
    absolute = std::max(absolute, gap);
    worst = std::max(worst, gap / resolution_of(expected[i]));
  }
  if (worst_absolute) *worst_absolute = absolute;
  return worst;
}

// ----------------------------------------------------------------------------------------------
// the flow
// ----------------------------------------------------------------------------------------------

int run_flow(std::size_t D, int rounds, double over_tc)
{
  double const K = 1.0 / (over_tc * reference::critical_temperature);
  double const onsager = onsager_lnz(K);
  std::printf("dense HOTRG, 2D Ising   T = %.6g Tc   K = %.16g   bond %zu   rounds %d\n", over_tc, K, D, rounds);
  std::printf("  the exact strip values are tabulated for T/Tc = 1 and 1.2 (ising_strip_reference.hpp)\n");
  std::printf("%6s %5s %9s %18s %11s %10s %10s %10s %10s\n", "sites", "bond", "cut", "ln Z / N", "vs Onsager", "X_1",
              "exact", "X_2", "exact");

  flow_record const record = flow(vertex(model::clock, 2, K), D, rounds, [&](flow_record const& r, tensor4 const& T0) {
    std::size_t const i = r.L.size() - 1;
    long long const sites = r.L[i];
    std::printf("%6lld %5d %9.2e %18.15f %11.2e", sites, static_cast<int>(T0.extent(0)),
                std::max(r.discarded_y[i], r.discarded_x[i]), r.lnz[i], (r.lnz[i] - onsager) / onsager);
    auto const X = scaling_dimensions(ring_levels(T0).eigenvalues, sites, sites);
    auto const* expected = exact_strip(sites, over_tc);
    if (X.size() >= 3)
    {
      if (expected)
        std::printf(" %10.6f %10.6f %10.6f %10.6f", X[1], expected->scaling_dimensions[1], X[2],
                    expected->scaling_dimensions[2]);
      else
        std::printf(" %10.6f %10s %10.6f %10s", X[1], "-", X[2], "-");
    }
    std::printf("\n");
  });

  double all = 0.0, contract = 0.0, regroup = 0.0, svd = 0.0;
  for (std::size_t i = 0; i < record.seconds.size(); ++i)
  {
    all += record.seconds[i];
    contract += record.seconds_contract[i];
    regroup += record.seconds_regroup[i];
    svd += record.seconds_svd[i];
  }
  std::printf("%.2f s: contractions (GEMM) %.2f, axis regroupings %.2f, SVD %.2f\n", all, contract, regroup, svd);
  std::printf("'vs Onsager' compares the L x L torus with the infinite lattice; the critical Ising CFT has "
              "X_1 = 1/8, X_2 = 1\n");
  return 0;
}

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
  std::printf("  %s: kept %d of %d, |U^T U - 1| %.1e, |theta U - U S| %.1e, discarded %.6f (definition %.6f)\n", name,
              static_cast<int>(k), static_cast<int>(n * n), orthonormal, eigen, step.discarded, discarded);
  report(orthonormal < 1e-12 && eigen < 1e-11 && descending && std::abs(discarded - step.discarded) < 1e-12,
         std::string(name) + ": U is the leading eigenbasis of the gram evaluated from the definition");
}

void kernel_against_the_definition()
{
  std::printf("\n== A  the kernel against the definition (random tensor, vertical dimension 2, horizontal 3) ==\n");
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
    std::printf("  gram: max |library - definition| along y %.1e, along x %.1e\n", worst_y, worst_x);
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
    std::printf(
        "  update network, operands (2,3,3,4) over (4,3,3,5), a random 3x3x4 U: max |library - definition| %.1e\n",
        worst);
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
  std::printf("\n== B  untruncated flows against enumerated partition functions ==\n");
  struct row
  {
      model which;
      char const* name;
      int q;
      double K;
      int steps;
      std::size_t D;
  };
  for (row const& r :
       {row{model::clock, "Ising", 2, 0.30, 2, 16}, row{model::clock, "Ising", 2, 0.4406867935, 2, 16},
        row{model::clock, "3-state clock", 3, 0.55, 1, 9}, row{model::potts, "3-state Potts", 3, 0.9, 1, 9},
        row{model::clock, "4-state clock", 4, 0.7, 1, 16}})
  {
    flow_record const record = flow(vertex(r.which, r.q, r.K), r.D, r.steps);
    double worst = std::abs(std::log(record.trace_T_unit) - enumerated_lnz(r.which, r.q, r.K, 1));
    std::printf("  %-14s K %.10f: L 1 %.1e", r.name, r.K, worst);
    for (int step = 0; step < r.steps; ++step)
    {
      int const L = 2 << step;
      double const exact = enumerated_lnz(r.which, r.q, r.K, L);
      double const gap = std::abs(record.lnz[static_cast<std::size_t>(step)] * L * L - exact) / exact;
      std::printf(", L %d %.1e", L, gap);
      worst = std::max(worst, gap);
    }
    std::printf("   relative error of ln Z\n");
    report(worst < 1e-12, std::string(r.name) + ": ln Z of the torus equals the sum over spin configurations");
  }
}

void clock_is_potts_at_q3()
{
  std::printf("\n== C  the q = 3 clock model at K is the 3-state Potts model at 3K/2 ==\n");
  double const K = 0.6;
  flow_record const clock = flow(vertex(model::clock, 3, K), 9, 8),
                    potts = flow(vertex(model::potts, 3, 1.5 * K), 9, 8);
  double const gap = std::abs(clock.lnz.back() - (potts.lnz.back() - K));
  std::printf("  D 9, L 256: ln Z / N clock %.14f, Potts - K %.14f, difference %.1e\n", clock.lnz.back(),
              potts.lnz.back() - K, gap);
  report(gap < 1e-12, "the two models' free energies differ by exactly K per site");
}

void onsager()
{
  std::printf("\n== D  Onsager's free energy (bond 8, L = 4096) ==\n");
  // What is left is the truncation: about 3e-6 away from the critical point and 1.4e-5 at it.
  for (auto const& [K, bound] : {std::pair{0.40, 1e-5}, std::pair{1.0 / reference::critical_temperature, 5e-5}})
  {
    flow_record const record = flow(vertex(model::clock, 2, K), 8, 12);
    double const exact = onsager_lnz(K), error = std::abs(record.lnz.back() - exact) / exact;
    std::printf("  K %.10f: ln Z / N %.12f, Onsager %.12f, relative error %.1e (bound %.0e)\n", K, record.lnz.back(),
                exact, error, bound);
    report(error < bound, "the free energy is Onsager's to the accuracy of a bond of 8");
  }
}

// ----------------------------------------------------------------------------------------------
// E  dispatch
// ----------------------------------------------------------------------------------------------

void every_contraction_in_gemm()
{
  std::printf("\n== E  where the contractions of one round run ==\n");
  flow_record const plateau = flow(vertex(model::clock, 2, 1.0 / reference::critical_temperature), 12, 6);
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
  for (auto const& [walk, count] : tally)
    std::printf("    %-10s -> %-28s x%d\n", walk.first.c_str(), walk.second.c_str(), count);
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
  std::printf("\n== F  the coarse tensor's transfer matrix against the exact Ising strip ==\n");
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
      tensor4 t = vertex(model::clock, 2, 1.0 / (over_tc * reference::critical_temperature));
      for (int step = 0; step < rounds; ++step)
        t = one_round(t, no_truncation);

      auto const spectrum = ring_levels(t);
      auto const X = scaling_dimensions(spectrum.eigenvalues, sites, sites);
      std::size_t const compared = std::min(X.size(), expected->count);
      double absolute = 0.0;
      double const relative = worst_relative_to_resolution(X, expected->scaling_dimensions, compared, &absolute);
      std::printf("  %d round%s -> %3lld sites at %.1f Tc: bond %3d, all %2zu levels compared, "
                  "max |X - exact| %.1e = %.2f of what is resolvable\n",
                  rounds, rounds == 1 ? " " : "s", sites, over_tc, static_cast<int>(t.extent(0)), compared, absolute,
                  relative);
      report(spectrum.asymmetry < 1e-12, "the transfer matrix of a reflection-symmetric tensor is symmetric");
      report(compared == expected->count && relative < 1.0, "its scaling dimensions are the exactly solved strip's");
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
  std::printf("\n== G  a truncated flow read width by width (bond 8, the lowest %zu levels) ==\n", lowest_levels);
  std::size_t const D = 8;
  for (double over_tc : {1.0, 1.2})
  {
    std::printf("  %.1f Tc: %6s %10s %10s %10s %10s %12s\n", over_tc, "sites", "X_1", "exact", "X_2", "exact", "worst");
    tensor4 t = vertex(model::clock, 2, 1.0 / (over_tc * reference::critical_temperature));
    double exact_while_wide = 0.0, last_first = 0.0, first_at_16 = 0.0, second_at_16 = 0.0;
    long long last_sites = 0;
    for (int rounds = 1; rounds <= 6; ++rounds)
    {
      bool const still_exact = static_cast<std::size_t>(t.extent(0)) * static_cast<std::size_t>(t.extent(0)) <= D;
      t = one_round(t, D);
      long long const sites = 1LL << rounds;
      auto const X = scaling_dimensions(ring_levels(t).eigenvalues, sites, sites);
      auto const* expected = exact_strip(sites, over_tc);
      if (X.size() < lowest_levels || !expected) continue;
      double absolute = 0.0;
      worst_relative_to_resolution(X, expected->scaling_dimensions, lowest_levels, &absolute);
      std::printf("  %7s %6lld %10.6f %10.6f %10.6f %10.6f %12.1e\n", "", sites, X[1], expected->scaling_dimensions[1],
                  X[2], expected->scaling_dimensions[2], absolute);
      if (still_exact) exact_while_wide = std::max(exact_while_wide, absolute);
      if (sites == 16)
      {
        first_at_16 = X[1];
        second_at_16 = X[2];
      }
      last_first = X[1];
      last_sites = sites;
    }
    report(exact_while_wide < 1e-8, "while the bond is wide enough the reading is still the exact strip's");
    if (over_tc == 1.0)
    {
      std::printf("  %7s the critical Ising CFT has X_sigma = 1/8 and X_eps = 1\n", "");
      report(std::abs(first_at_16 - 0.125) < 0.005 && std::abs(second_at_16 - 1.0) < 0.08,
             "at T_c the reading at 16 sites is near the Ising CFT dimensions");
    }
    else
    {
      std::printf("  %7s above T_c the strip is disordered, so X grows with the width\n", "");
      report(last_first > 1.0 && last_sites >= 32, "above T_c the reading leaves the critical values");
    }
  }
}

// ----------------------------------------------------------------------------------------------
// H  inputs at the edge
// ----------------------------------------------------------------------------------------------

void inputs_at_the_edge()
{
  std::printf("\n== H  inputs at the edge ==\n");

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
    std::printf("  rank-one gram, 2 states kept: singular values %.3e and %.3e, gap %g\n", step.S.front(),
                step.S.back(), step.gap);
    report(step.S.size() == 2 && std::isfinite(step.gap) && step.gap >= 0.0 && step.gap <= 1.0,
           "a cut inside the null space of the gram reports a finite gap");
  }

  // The width 2^rounds is a long long: 62 rounds fit, 63 do not.  A bond of 1 keeps this cheap.
  {
    tensor4 const unit = vertex(model::clock, 2, 0.4);
    flow_record const widest = flow(unit, 1, max_rounds);
    std::printf("  %d rounds: width %lld\n", max_rounds, widest.L.back());
    report(widest.L.back() == (1LL << max_rounds), "the largest round count is carried without overflow");
    bool refused = false;
    try
    {
      static_cast<void>(flow(unit, 1, max_rounds + 1));
    }
    catch (std::invalid_argument const&)
    {
      refused = true;
    }
    report(refused, "one round more is refused, because its width would not fit");
  }
}

int run_checks()
{
  std::printf("dense HOTRG checks\n");
  kernel_against_the_definition();
  exact_tori();
  clock_is_potts_at_q3();
  onsager();
  every_contraction_in_gemm();
  transfer_matrix_against_the_exact_strip();
  truncated_flow_against_the_exact_strip();
  inputs_at_the_edge();
  std::printf("\n%s  (%d failed)\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED", failures);
  return failures == 0 ? 0 : 1;
}

// ----------------------------------------------------------------------------------------------
// arguments
// ----------------------------------------------------------------------------------------------

static_assert(max_rounds == 62, "the usage text and the --rounds complaint state this limit");

[[noreturn]] void usage(char const* complaint)
{
  if (complaint) std::fprintf(stderr, "ising_hotrg_example: %s\n", complaint);
  std::fprintf(stderr, "usage: ising_hotrg_example [--bond=D] [--rounds=N] [--temperature=T/Tc] [--check]\n"
                       "  --bond=D          bond dimension kept by the cut (default 16)\n"
                       "  --rounds=N        HOTRG rounds; the lattice is 2^N x 2^N (default 8, at most 62)\n"
                       "  --temperature=X   temperature in units of the exact Tc (default 1)\n"
                       "  --check           verify the implementation instead of running a flow\n");
  std::exit(complaint ? 2 : 0);
}

[[nodiscard]] std::string_view option_value(std::string_view argument, std::string_view prefix)
{
  if (!argument.starts_with(prefix)) return {};
  return argument.substr(prefix.size());
}

template <class Integer> [[nodiscard]] Integer parse_integer(std::string_view text, char const* name)
{
  Integer value{};
  auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value < 1)
    usage((std::string(name) + " takes a positive integer").c_str());
  return value;
}

} // namespace

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::size_t D = 16;
  int rounds = 8;
  double over_tc = 1.0;
  bool check = false;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view const argument = argv[i];
    if (argument == "--help" || argument == "-h")
      usage(nullptr);
    else if (argument == "--check")
      check = true;
    else if (auto const value = option_value(argument, "--bond="); !value.empty())
      D = parse_integer<std::size_t>(value, "--bond");
    else if (auto const value = option_value(argument, "--rounds="); !value.empty())
    {
      rounds = parse_integer<int>(value, "--rounds");
      if (rounds > max_rounds) usage("--rounds is at most 62: the lattice width 2^rounds is kept in 64 bits");
    }
    else if (auto const value = option_value(argument, "--temperature="); !value.empty())
    {
      over_tc = std::strtod(std::string(value).c_str(), nullptr);
      if (!(over_tc > 0.0)) usage("--temperature takes a positive number");
    }
    else
      usage(("unknown argument " + std::string(argument)).c_str());
  }
  try
  {
    return check ? run_checks() : run_flow(D, rounds, over_tc);
  }
  catch (std::exception const& error)
  {
    std::fprintf(stderr, "ising_hotrg_example: %s\n", error.what());
    return 1;
  }
}
