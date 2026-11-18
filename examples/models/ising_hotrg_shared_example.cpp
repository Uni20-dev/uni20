// ising_hotrg_shared_example.cpp -- one HOTRG step for host tensors and for CUDA tensors.
//
// dense_hotrg_shared.hpp writes the step of dense_hotrg.hpp once, as templates over the place
// where the tensors live.  This program runs it in either place:
//
//   ising_hotrg_shared_example [--execution=cpu|cuda] [--cuda-device=N] [--gram-cut=device|host]
//                              [--bond=D] [--rounds=N] [--temperature=T/Tc] [--reference] [--check]
//
// Without --check it runs one flow of the square-lattice Ising model and prints, round by round,
// the wall time, where it went, ln Z per site and the discarded weight, and then which backend
// ran each kind of operation.  The first rounds grow the bond (2, 4, 16, ...) and are cheap; the
// rounds at the full bond are the ones to read.
//
// With --check it compares the shared step with the step of dense_hotrg.hpp and exits non-zero
// on any failure:
//
//   A  the gram of a tensor with unequal extents, entry by entry
//   B  the update network with a given (arbitrary) isometry, entry by entry
//   C  one step along y and along x: the kept spectrum, the discarded weight, the gap, and the one
//      number of the coarse tensor that does not depend on the signs of the singular vectors
//   D  two flows of the critical Ising model: ln Z per site and the cut errors, round by round
//
// --execution=cpu checks the host place.  --execution=cuda checks the host place, the device
// place, and the device place with the gram cut on the host.  The coarse tensors are not compared
// entry by entry after an SVD: two SVD codes may return a singular vector with either sign, which
// changes the tensor but nothing read from it.
#include "dense_hotrg_shared.hpp"

#include <uni20/linalg/dispatch_diagnostics.hpp>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace uni20::examples::dense_hotrg;
namespace shared = uni20::examples::dense_hotrg_shared;

namespace
{

// ----------------------------------------------------------------------------------------------
// checks
// ----------------------------------------------------------------------------------------------

int failures = 0;

void report(bool ok, std::string const& what)
{
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

template <class Tensor> void fill(Tensor& t, double seed)
{
  double angle = seed;
  for (auto& value : t.storage())
    value = std::sin(angle += 1.0);
}

// A host tensor in the place under test, and a tensor of that place back on the host.
template <class Tensor> [[nodiscard]] Tensor const& there(shared::host_place, Tensor const& t) { return t; }
template <class Tensor> [[nodiscard]] Tensor const& back(shared::host_place, Tensor const& t) { return t; }
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
template <class Tensor> [[nodiscard]] auto there(shared::device_place, Tensor const& t)
{
  return uni20::to_device(t, uni20::cuda::device_resources());
}
template <class Tensor> [[nodiscard]] auto back(shared::device_place, Tensor const& t) { return uni20::to_host(t); }
#endif

template <class A, class B> double relative_difference(A const& a, B const& b)
{
  double worst = 0.0, scale = 0.0;
  auto other = b.storage().begin();
  for (auto const& value : a.storage())
  {
    scale = std::max(scale, std::abs(static_cast<double>(value)));
    worst = std::max(worst, std::abs(static_cast<double>(value) - static_cast<double>(*other)));
    ++other;
  }
  return worst / scale;
}

double largest_difference(std::vector<double> const& a, std::vector<double> const& b)
{
  double worst = a.size() == b.size() ? 0.0 : HUGE_VAL;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

template <class Place> void kernels(std::string const& where)
{
  std::printf("\n== A, B  the gram and the update network on %s, entry by entry ==\n", where.c_str());
  tensor4 t(3, 4, 4, 3);
  fill(t, 0.3);
  auto const& t_there = there(Place{}, t);
  auto const gram_there = shared::gram_y(t_there);
  auto const& gram_back = back(Place{}, gram_there);
  double const gram_difference = relative_difference(gram_y(t), gram_back);
  std::printf("  gram, extents (3, 4, 4, 3): off dense_hotrg.hpp by %.2e of its largest entry\n", gram_difference);
  report(gram_difference < 1e-13, "the gram on " + where + " is that of dense_hotrg.hpp");

  tensor4 upper(3, 4, 4, 5), lower(5, 4, 4, 2);
  tensor3 iso(4, 4, 6);
  fill(upper, 0.1);
  fill(lower, 0.5);
  fill(iso, 0.9);
  auto const& upper_there = there(Place{}, upper);
  auto const& lower_there = there(Place{}, lower);
  auto const& iso_there = there(Place{}, iso);
  auto const update_there = shared::apply_isometry(upper_there, lower_there, iso_there);
  auto const& update_back = back(Place{}, update_there);
  double const update_difference = relative_difference(apply_isometry(upper, lower, iso), update_back);
  std::printf("  update network, upper (3, 4, 4, 5), lower (5, 4, 4, 2), isometry (4, 4, 6): off by %.2e\n",
              update_difference);
  report(update_difference < 1e-13, "the update network on " + where + " is that of dense_hotrg.hpp");
}

template <class Place> void one_step(shared::device_svd route, std::string const& where)
{
  std::printf("\n== C  one step on %s ==\n", where.c_str());
  double const K = 1.0 / critical_temperature(model::clock, 2);
  flow_record const start = flow(vertex(model::clock, 2, K), 4, 2); // a generic 4^4 tensor
  tensor4 const& t = start.T0;
  auto const& t_there = there(Place{}, t);
  for (axis along : {axis::y, axis::x})
  {
    char const* const name = along == axis::y ? "y" : "x";
    step_result const expected = coarse_grain(t, along, 6);
    double const invariant = expected.norm * torus_trace(expected.T);
    auto const step = shared::coarse_grain(t_there, along, 6, route);
    auto const& coarse = back(Place{}, step.T);
    double const spectrum = largest_difference(expected.S, step.S) / expected.S.front();
    double const weight = std::abs(expected.discarded - step.discarded);
    double const gap = std::abs(expected.gap - step.gap);
    double const trace = std::abs(step.norm * torus_trace(coarse) - invariant) / std::abs(invariant);
    std::printf("  %s step: spectrum %.1e, discarded weight %.1e, gap %.1e, norm x trace %.1e\n", name, spectrum,
                weight, gap, trace);
    report(spectrum < 1e-12 && weight < 1e-12 && gap < 1e-9 && trace < 1e-12,
           std::string("the ") + name + " step on " + where + " agrees with dense_hotrg.hpp");
  }
}

template <class Place> void flows(shared::device_svd route, std::string const& where)
{
  std::printf("\n== D  flows of the critical Ising model on %s ==\n", where.c_str());
  double const K = 1.0 / critical_temperature(model::clock, 2);
  tensor4 const unit = vertex(model::clock, 2, K);
  for (std::size_t D : {std::size_t{3}, std::size_t{8}})
  {
    int const rounds = 6;
    flow_record const expected = flow(unit, D, rounds);
    flow_record const got = shared::flow<Place>(unit, D, rounds, route);
    double const lnz = largest_difference(expected.lnz, got.lnz);
    double const cut = std::max(largest_difference(expected.discarded_y, got.discarded_y),
                                largest_difference(expected.discarded_x, got.discarded_x));
    std::printf("  bond %zu, %d rounds: ln Z / N %.15f, off dense_hotrg.hpp by %.1e; cut errors off by %.1e\n", D,
                rounds, got.lnz.back(), lnz, cut);
    report(lnz < 1e-11 && cut < 1e-9,
           "bond " + std::to_string(D) + ": the flow on " + where + " is that of dense_hotrg.hpp");
  }
}

template <class Place> void check_place(shared::device_svd route, std::string const& where, bool with_kernels)
{
  if (with_kernels) kernels<Place>(where);
  one_step<Place>(route, where);
  flows<Place>(route, where);
}

int run_checks(bool on_device)
{
  std::printf("one HOTRG step for host and CUDA tensors (dense_hotrg_shared.hpp), against dense_hotrg.hpp\n");
  check_place<shared::host_place>(shared::device_svd::cusolver, "host tensors", true);
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
  if (on_device)
  {
    check_place<shared::device_place>(shared::device_svd::cusolver, "CUDA tensors", true);
    check_place<shared::device_place>(shared::device_svd::host, "CUDA tensors with the gram cut on the host", false);
  }
#else
  static_cast<void>(on_device);
#endif
  std::printf("\n%s  (%d failed)\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED", failures);
  return failures == 0 ? 0 : 1;
}

// ----------------------------------------------------------------------------------------------
// one flow, timed
// ----------------------------------------------------------------------------------------------

struct options
{
    std::size_t D = 16;
    int rounds = 6;
    double over_tc = 1.0;
    bool on_device = false;
    int cuda_device = 0;
    shared::device_svd gram_cut = shared::device_svd::cusolver;
    bool reference = false;
    bool check = false;
};

template <class Place> int run_flow(options const& run)
{
  double const K = 1.0 / (run.over_tc * critical_temperature(model::clock, 2));
  tensor4 const unit = vertex(model::clock, 2, K);
  char const* const step = run.reference ? "dense_hotrg.hpp" : "dense_hotrg_shared.hpp";
  char const* const cut = !run.on_device                             ? ""
                          : run.gram_cut == shared::device_svd::host ? ", gram cut on the host"
                                                                     : ", gram cut on the device";
  std::printf("dense HOTRG, 2D Ising, T = %g Tc, bond %zu, %d rounds, step of %s on %s tensors%s\n", run.over_tc, run.D,
              run.rounds, step, run.on_device ? "CUDA" : "host", cut);
  std::printf("%6s %5s %10s %10s %10s %10s %10s  %-20s %s\n", "round", "bond", "seconds", "contract", "regroup", "svd",
              "via host", "ln Z / N", "cut (x step)");

  // Which backend ran each kind of operation, counted over the whole run.
  std::map<std::string, long> backends;
  uni20::linalg::dispatch_diagnostics::scoped_sink const watch([&backends](auto const& event) {
    if (event.selected_backend())
      ++backends[std::string(event.operation) + " -> " + std::string(*event.selected_backend())];
  });

  // "via host" is the time the device place spends in the stand-ins that download a tensor
  // (shim::largest_entry and shim::torus_trace); it is zero on the host.
  double via_host_before = shared::spent().fallback;
  auto observe = [&](flow_record const& r, tensor4 const& T0) {
    std::size_t const i = r.L.size() - 1;
    double const via_host = shared::spent().fallback - via_host_before;
    via_host_before = shared::spent().fallback;
    std::printf("%6zu %5d %10.3f %10.3f %10.3f %10.3f %10.3f  %.15f  %.2e\n", i + 1, static_cast<int>(T0.extent(0)),
                r.seconds[i], r.seconds_contract[i], r.seconds_regroup[i], r.seconds_svd[i], via_host, r.lnz[i],
                r.discarded_x[i]);
  };
  if (run.reference)
    static_cast<void>(flow(unit, run.D, run.rounds, observe));
  else
    static_cast<void>(shared::flow<Place>(unit, run.D, run.rounds, run.gram_cut, observe));
  std::printf("backends:");
  for (auto const& [what, count] : backends)
    std::printf("  %s (%ld)", what.c_str(), count);
  std::printf("\n");
  return 0;
}

// ----------------------------------------------------------------------------------------------
// arguments
// ----------------------------------------------------------------------------------------------

[[noreturn]] void usage(char const* complaint)
{
  if (complaint) std::fprintf(stderr, "ising_hotrg_shared_example: %s\n", complaint);
  std::fprintf(
      stderr, "usage: ising_hotrg_shared_example [--execution=cpu|cuda] [--cuda-device=N] [--gram-cut=device|host]\n"
              "                                  [--bond=D] [--rounds=N] [--temperature=T/Tc] [--reference] [--check]\n"
              "  --execution=cpu|cuda    where the tensors live (default cpu); cuda needs the CUDA, cuBLAS and\n"
              "                          cuSOLVER backends\n"
              "  --cuda-device=N         device ordinal for --execution=cuda (default 0)\n"
              "  --gram-cut=device|host  with --execution=cuda: decompose the gram on the device (default) or\n"
              "                          download it and decompose it on the host\n"
              "  --bond=D                bond dimension kept by the cut (default 16)\n"
              "  --rounds=N              HOTRG rounds; the lattice is 2^N x 2^N (default 6, at most 62)\n"
              "  --temperature=X         temperature in units of the exact Tc (default 1)\n"
              "  --reference             run the step of dense_hotrg.hpp instead, for comparison (cpu only)\n"
              "  --check                 compare the shared step with dense_hotrg.hpp instead of running a flow\n");
  std::exit(complaint ? 2 : 0);
}

[[nodiscard]] std::string_view option_value(std::string_view argument, std::string_view prefix)
{
  if (!argument.starts_with(prefix)) return {};
  return argument.substr(prefix.size());
}

template <class Integer> [[nodiscard]] Integer parse_integer(std::string_view text, char const* name, Integer least)
{
  Integer value{};
  auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value < least)
    usage((std::string(name) + " takes an integer, at least " + std::to_string(least)).c_str());
  return value;
}

[[nodiscard]] options parse_options(int argc, char** argv)
{
  options run;
  bool gram_cut_given = false;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view const argument = argv[i];
    if (argument == "--help" || argument == "-h")
      usage(nullptr);
    else if (argument == "--check")
      run.check = true;
    else if (argument == "--reference")
      run.reference = true;
    else if (argument == "--execution=cpu")
      run.on_device = false;
    else if (argument == "--execution=cuda")
      run.on_device = true;
    else if (argument == "--gram-cut=device" || argument == "--gram-cut=host")
    {
      run.gram_cut = argument == "--gram-cut=host" ? shared::device_svd::host : shared::device_svd::cusolver;
      gram_cut_given = true;
    }
    else if (auto const value = option_value(argument, "--cuda-device="); !value.empty())
      run.cuda_device = parse_integer<int>(value, "--cuda-device", 0);
    else if (auto const value = option_value(argument, "--bond="); !value.empty())
      run.D = parse_integer<std::size_t>(value, "--bond", 1);
    else if (auto const value = option_value(argument, "--rounds="); !value.empty())
    {
      run.rounds = parse_integer<int>(value, "--rounds", 1);
      if (run.rounds > max_rounds) usage("--rounds is at most 62: the lattice width 2^rounds is kept in 64 bits");
    }
    else if (auto const value = option_value(argument, "--temperature="); !value.empty())
    {
      run.over_tc = std::strtod(std::string(value).c_str(), nullptr);
      if (!(run.over_tc > 0.0)) usage("--temperature takes a positive number");
    }
    else
      usage(("unknown argument " + std::string(argument)).c_str());
  }
  if (run.reference && run.on_device) usage("--reference runs the step of dense_hotrg.hpp, which is host only");
  if (gram_cut_given && !run.on_device) usage("--gram-cut applies to --execution=cuda");
  return run;
}

static_assert(max_rounds == 62, "the usage text and the --rounds complaint state this limit");

} // namespace

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  options const run = parse_options(argc, argv);
  try
  {
    if (run.on_device)
    {
#if UNI20_EXAMPLE_DENSE_HOTRG_DEVICE
      auto cuda_lifetime = uni20::cuda::initialize(
          {.device_ordinals = {run.cuda_device}, .default_device = run.cuda_device, .streams_per_device = 4});
      uni20::cuda::ScopedDevice selected_device(run.cuda_device);
      return run.check ? run_checks(true) : run_flow<shared::device_place>(run);
#else
      throw std::runtime_error("--execution=cuda requires the CUDA, cuBLAS and cuSOLVER backends");
#endif
    }
    return run.check ? run_checks(false) : run_flow<shared::host_place>(run);
  }
  catch (std::exception const& error)
  {
    std::fprintf(stderr, "ising_hotrg_shared_example: %s\n", error.what());
    return 1;
  }
}
