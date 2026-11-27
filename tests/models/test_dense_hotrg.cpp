#include "../../examples/models/dense_hotrg.hpp"
#include "../../examples/models/ising_hotrg_output.hpp"
#include "../../examples/models/ising_strip_reference.hpp"
#include "../numerics/precision_cases.hpp"
#include <sstream>
#include <uni20/linalg/dispatch_diagnostics.hpp>

namespace
{
namespace hotrg = uni20::examples::dense_hotrg;
using namespace uni20;
using namespace uni20::test;

using Cases = testing::Types<PrecisionCase<PrecisionKind::float32, false>, PrecisionCase<PrecisionKind::float64, false>
#if UNI20_HAS_MPLAPACK_FLOAT80
                             ,
                             PrecisionCase<PrecisionKind::float80, false>
#endif
#if UNI20_HAS_MPLAPACK_FLOAT128
                             ,
                             PrecisionCase<PrecisionKind::float128, false>
#endif
#if UNI20_HAS_MPLAPACK_MPREAL
                             ,
                             PrecisionCase<PrecisionKind::mp128, false>, PrecisionCase<PrecisionKind::mp256, false>
#endif
                             >;
template <class C> class DenseHotrg : public PrecisionTest<C> {};
TYPED_TEST_SUITE(DenseHotrg, Cases);

TYPED_TEST(DenseHotrg, ExactTorusFromSpinConfigurations)
{
  using C = TypeParam;
  using R = typename C::real_type;
  auto const p = C::precision();
  int contractions = 0;
  linalg::dispatch_diagnostics::scoped_sink const watch([&](auto const& event) {
    EXPECT_TRUE(event.succeeded());
    auto const selected = event.selected_backend();
    if (!selected) return;
    testing::Test::RecordProperty(event.operation, std::string(*selected));
    if (event.operation == "gemm")
    {
      EXPECT_EQ(*selected, "blas");
    }
    if (event.operation == "contract")
    {
      ++contractions;
      EXPECT_NE(selected->find("gemm"), std::string_view::npos);
    }
  });
  R const K = C::real(3) / 10;
  auto const record = hotrg::flow(hotrg::vertex(hotrg::model::clock, 2, K), 4, 1);
  R z = C::real(0);
  for (int bits = 0; bits < 16; ++bits)
  {
    int energy = 0;
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
      {
        auto spin = [&](int site) { return bits & (1 << site) ? 1 : -1; };
        energy += spin(2 * y + x) * (spin(2 * y + (1 - x)) + spin(2 * (1 - y) + x));
      }
    z += math::exp(K * (energy - 8));
  }
  R const expected = 2 * K + math::log(z) / 4;
  expect_error_at_most(record.lnz.back(), expected, 128 * C::epsilon());
  expect_error_at_most(record.unit_lnz, 2 * K + math::log(C::real(2)), 16 * C::epsilon());
  C::expect_precision(record.lnz.back());
  EXPECT_EQ(contractions, 12);
  EXPECT_FALSE(hotrg::ring_levels(record.T0).eigenvalues.empty());
  EXPECT_EQ(precision_of(record.T0), p);
  for (auto const& value : record.T0.storage())
    C::expect_precision(value);
}

TYPED_TEST(DenseHotrg, ExactStripUsesAllAvailableDigits)
{
  using C = TypeParam;
  using R = typename C::real_type;
  auto const p = C::precision();
  R const Kc = hotrg::critical_coupling<R>(hotrg::model::clock, 2, p);
  for (int numerator : {5, 6})
  {
    R const ratio = C::real(numerator) / 5;
    auto const record =
        hotrg::flow(hotrg::vertex(hotrg::model::clock, 2, R(Kc / ratio)), 16, 2, [&](auto const& r, auto const& t) {
          auto const spectrum = hotrg::ring_levels(t);
          auto const X = hotrg::scaling_dimensions(spectrum.eigenvalues);
          auto const* reference = hotrg::reference::find(r.log2_width.back(), ratio);
          ASSERT_NE(reference, nullptr);
          ASSERT_GE(X.size(), 3u);
          ASSERT_LE(X.size(), reference->scaling_dimensions.size());
          R const pi = hotrg::detail::pi_value<R>(p);
          for (std::size_t i = 0; i < X.size(); ++i)
          {
            R const expected = hotrg::reference::value<R>(reference->scaling_dimensions[i], p);
            R const bound = 128 * C::epsilon() * math::exp(2 * pi * expected);
            expect_error_at_most(X[i], expected, bound);
            C::expect_precision(X[i]);
          }
          expect_error_at_most(spectrum.asymmetry, C::real(0), 128 * C::epsilon());
        });
    EXPECT_EQ(record.log2_width.back(), 2);
  }
}

TYPED_TEST(DenseHotrg, EighCutKeepsTheSvdCutSubspace)
{
  using C = TypeParam;
  using R = typename C::real_type;
  auto const p = C::precision();
  R const Kc = hotrg::critical_coupling<R>(hotrg::model::clock, 2, p);
  // At bond 8 the gap at the cut is wide, so both methods keep the same subspace and the flows agree
  // to rounding, round by round.
  auto const svd = hotrg::flow(hotrg::vertex(hotrg::model::clock, 2, Kc), 8, 3);
  auto const eigh = hotrg::flow(hotrg::vertex(hotrg::model::clock, 2, Kc), 8, 3, nullptr, hotrg::cut_method::eigh);
  ASSERT_EQ(eigh.lnz.size(), svd.lnz.size());
  for (std::size_t i = 0; i < svd.lnz.size(); ++i)
  {
    expect_error_at_most(eigh.lnz[i], svd.lnz[i], 1024 * C::epsilon());
    expect_error_at_most(eigh.discarded_y[i], svd.discarded_y[i], 1024 * C::epsilon());
    expect_error_at_most(eigh.discarded_x[i], svd.discarded_x[i], 1024 * C::epsilon());
    C::expect_precision(eigh.lnz[i]);
  }
}

TYPED_TEST(DenseHotrg, LongFlowRetainsTinyLogScale)
{
  using C = TypeParam;
  using R = typename C::real_type;
  R const gap = C::gap();
  hotrg::tensor4<R> tensor(1, 1, 1, 1, C::precision());
  tensor[0, 0, 0, 0] = C::real(1) + gap;
  auto const record = hotrg::flow(hotrg::scaled_tensor<R>{std::move(tensor), C::real(0)}, 1, 128);
  EXPECT_EQ(record.log2_width.front(), 1);
  EXPECT_EQ(record.log2_width.back(), 128);
  R const expected = math::log1p(gap);
  expect_error_at_most(record.lnz.back(), expected, 8 * C::epsilon() * gap);
  C::expect_precision(record.initial_log_scale);
  C::expect_precision(record.log_norm_y.back());
  // Large but equal dimensions cancel in the exponent before any scalar scaling.
  std::vector<R> eigenvalues{C::real(1), C::real(1) - gap};
  auto const X = hotrg::scaling_dimensions(eigenvalues, 1000000, 1000000);
  R const target = -math::log1p(-gap) / (2 * hotrg::detail::pi_value<R>(C::precision()));
  expect_error_at_most(X[1], target, 8 * C::epsilon() * gap);
}

TYPED_TEST(DenseHotrg, InitialTensorIsNormalizedBeforeGram)
{
  using C = TypeParam;
  using R = typename C::real_type;
  hotrg::tensor4<R> tensor(1, 1, 1, 1, C::precision());
  // The first Gram is quartic: this finite input would overflow it in every fixed type.
  int const exponent = std::same_as<R, float> ? 100 : std::same_as<R, double> ? 300 : 5000;
  tensor[0, 0, 0, 0] = math::scalbn(C::real(1), exponent);
  auto const record = hotrg::flow(hotrg::scaled_tensor<R>{std::move(tensor), C::real(0)}, 1, 2);
  R const expected = exponent * math::log(C::real(2));
  expect_error_at_most(record.lnz.back(), expected, 128 * C::epsilon() * expected);
}

TYPED_TEST(DenseHotrg, LowTemperatureUsesBoundedWeights)
{
  using C = TypeParam;
  using R = typename C::real_type;
  R const K = C::real(1000);
  auto const record = hotrg::flow(hotrg::vertex(hotrg::model::clock, 2, K), 4, 2);
  // At this temperature all non-ground configurations are below the tested precision.
  R const expected = 2 * K + math::log(C::real(2)) / 16;
  expect_error_at_most(record.lnz.back(), expected, 128 * C::epsilon() * expected);
  expect_error_at_most(hotrg::onsager_lnz(K, 16), 2 * K, 16 * C::epsilon() * K);
  EXPECT_TRUE(isfinite(record.discarded_y.front()));
  C::expect_precision(record.discarded_y.front());
}

TYPED_TEST(DenseHotrg, OutputPreservesPrecisionAndMissingLevels)
{
  using C = TypeParam;
  using R = typename C::real_type;
  namespace p = presentation;
  R const value = C::real(1) + C::gap();
  auto results = hotrg::make_flow_table<R>(C::precision());
  std::ostringstream exported;
  results.attach(p::tsv_sink(exported, {.projection = {.columns = {"log2_width", "lnz_per_site", "x1"}}}));
  std::string displayed;
  auto policy = p::plain_policy();
  policy.glyphs = p::glyph_set::ascii;
  policy.charset = p::text_charset::ascii_escape;
  results.attach(p::terminal_sink({.wrap_width = 40,
                                   .output =
                                       [&](display::event const& event) {
                                         displayed += p::render(std::get<p::styled_text>(event.content), policy);
                                       },
                                   .policy = policy}));
  results.append(1100, index_type{4}, value, value, value, std::nullopt, std::nullopt, value, value, value);
  results.finish();

  std::istringstream rows(exported.str());
  std::string heading, row;
  std::getline(rows, heading);
  std::getline(rows, row);
  auto const first_tab = row.find('\t'), last_tab = row.rfind('\t');
  ASSERT_NE(first_tab, std::string::npos);
  ASSERT_GT(last_tab, first_tab);
  EXPECT_EQ(row.substr(0, first_tab), "1100");
  EXPECT_EQ(row.substr(last_tab + 1), ""); // A missing level remains an absent number.
  R const recovered = hotrg::reference::value<R>(row.substr(first_tab + 1, last_tab - first_tab - 1), C::precision());
  EXPECT_EQ(recovered, value);
  C::expect_precision(recovered);
  EXPECT_NE(displayed.find(format_scalar(value)), std::string::npos); // Narrow terminals keep numeric tokens intact.
  EXPECT_EQ(results.retained_size(), 0u);
}

#if UNI20_HAS_MPLAPACK_MPREAL
TEST(DenseHotrgOutput, ReferenceDigitsArePreservedAboveFixturePrecision)
{
  namespace p = presentation;
  auto const source = hotrg::reference::strips[0].scaling_dimensions[1];
  auto const expected = parse_real<mpreal>(source, Precision::exact());
  for (int bits : {333, 512})
  {
    auto const precision = Precision::bits(bits);
    auto const reference = hotrg::displayed_reference<mpreal>(source, precision);
    auto results = hotrg::make_flow_table<mpreal>(precision);
    std::ostringstream exported;
    results.attach(p::csv_sink(exported, {.projection = {.columns = {"exact_x1"}}}));
    auto const zero = parse_real<mpreal>("0", precision);
    results.append(1, index_type{1}, zero, zero, zero, std::nullopt, reference, std::nullopt, std::nullopt, zero);
    results.finish();
    std::istringstream rows(exported.str());
    std::string heading, row;
    std::getline(rows, heading);
    std::getline(rows, row);
    EXPECT_TRUE(reference.is_exact());
    EXPECT_EQ(parse_real<mpreal>(row, Precision::exact()), expected);
  }
}
#endif

TEST(DenseHotrgCoverage, ConfiguredScalars)
{
  testing::Test::RecordProperty("fp32", "supported: native LAPACK");
  testing::Test::RecordProperty("fp64", "supported: native LAPACK");
  testing::Test::RecordProperty("fp80", UNI20_HAS_MPLAPACK_FLOAT80 ? "supported: MPLAPACK"
                                                                   : "unavailable: MPLAPACK binary80 disabled");
  testing::Test::RecordProperty("fp128", UNI20_HAS_MPLAPACK_FLOAT128 ? "supported: MPLAPACK"
                                                                     : "unavailable: MPLAPACK binary128 disabled");
  testing::Test::RecordProperty("mpreal_128_256", UNI20_HAS_MPLAPACK_MPREAL
                                                      ? "supported: MPLAPACK GEMM and decompositions"
                                                      : "unavailable: MPLAPACK MPFR disabled");
  testing::Test::RecordProperty("complex", "inapplicable: real HOTRG example");
}
} // namespace
