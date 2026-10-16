#include <gtest/gtest.h>
#include <locale.h>
#include <sstream>
#include <uni20/common/data_table_sinks.hpp>
#include <uni20/common/output_session.hpp>

#if UNI20_ENABLE_MPFR
namespace
{
using namespace uni20;
namespace p = uni20::presentation;

#if UNI20_HAS_FLOAT128
TEST(MprealMetadata, Binary128ConversionPreservesPrecisionAndChecksOverflow)
{
  auto native = uni20::float128(1) + numeric_limits<uni20::float128>::epsilon();
  auto exact = mpreal{1} + pow(mpreal{2}, mpreal{-112});
  for (auto bits : {128, 256})
  {
    auto precision = Precision::bits(bits);
    auto imported = metadata_value(native).as<mpreal>(precision);
    EXPECT_EQ(imported.precision(), precision);
    EXPECT_EQ(imported, exact);
    EXPECT_TRUE(metadata_value(imported).as<uni20::float128>() == native);
    EXPECT_TRUE(metadata_value(exact).as<uni20::float128>() == native);
    EXPECT_EQ(metadata_value(std::optional<uni20::float128>(native)).as<mpreal>(precision), exact);
    EXPECT_EQ(metadata_value(native).as<std::optional<mpreal>>(precision)->precision(), precision);
    EXPECT_FALSE(metadata_value(std::optional<uni20::float128>{}).as<std::optional<mpreal>>(precision));
    EXPECT_THROW((void)metadata_value(mpreal("1e10000", precision)).as<uni20::float128>(), std::out_of_range);
    auto largest = numeric_limits<uni20::float128>::max();
    EXPECT_TRUE(metadata_value(mpreal(largest, precision)).as<uni20::float128>() == largest);
    auto overflow = mpreal(largest, precision) * 2;
    EXPECT_THROW((void)metadata_value(overflow).as<uni20::float128>(), std::out_of_range);
    EXPECT_THROW((void)metadata_value(-overflow).as<uni20::float128>(), std::out_of_range);
  }
  EXPECT_THROW((void)metadata_value(native).as<mpreal>(), std::invalid_argument);
  EXPECT_THROW((void)metadata_value(native).as<mpreal>(Precision::exact()), std::logic_error);
}
#endif

TEST(MprealDataTable, RetainsPerValuePrecisionAndRejectsUnsetBeforeDelivery)
{
  auto table = p::make_data_table("mixed", p::data_column<mpreal>("x"), p::data_column<std::optional<mpreal>>("y"));
  for (auto bits : {128, 256})
  {
    auto precision = Precision::bits(bits);
    auto value = mpreal{1}.at(precision) + numeric_limits<mpreal>::epsilon(precision);
    table.append(value, std::nullopt);
    auto const& stored = std::get<0>(table.rows().back());
    EXPECT_EQ(stored, value);
    EXPECT_EQ(stored.precision(), precision);
    EXPECT_NE(stored, mpreal(static_cast<long double>(stored), precision));
  }
  table.append(3, mpreal("1/3", Precision::exact()));
  EXPECT_TRUE(std::get<0>(table.rows().back()).is_exact());
  std::ostringstream live;
  table.attach(p::json_sink(live));
  auto before = live.str();
  EXPECT_THROW(table.append(mpreal(uninitialized), std::nullopt), std::invalid_argument);
  EXPECT_THROW(table.append(1, std::optional<mpreal>(mpreal(uninitialized))), std::invalid_argument);
  EXPECT_EQ(table.size(), 3U);
  EXPECT_EQ(live.str(), before);
  table.finish();
}

TEST(MprealDataTable, ExactFormattingIsExplicitAndJsonIsIndependent)
{
  auto table = p::make_data_table("rational", p::data_column<mpreal>("x"));
  table.append(mpreal("1/3", Precision::exact()));
  std::ostringstream ambiguous;
  EXPECT_THROW(p::write_csv(ambiguous, table), std::invalid_argument);
  auto fraction = p::data_column<mpreal>("x").fractional().display();
  auto decimal = p::data_column<mpreal>("x").fixed(6).display();
  std::ostringstream csv, tsv, json;
  p::write_csv(csv, table, {.projection = {.display = {{"x", fraction}}}});
  p::write_tsv(tsv, table, {.projection = {.display = {{"x", decimal}}}});
  p::write_json(json, table);
  EXPECT_EQ(csv.str(), "x\n1/3\n");
  EXPECT_EQ(tsv.str(), "x\n0.333333\n");
  EXPECT_NE(json.str().find("\"rows\":[[{\"value\":\"1/3\",\"precision\":\"exact\"}]]"), std::string::npos);
  EXPECT_EQ(std::get<0>(table.rows()[0]).to_string(), "1/3");
  p::report_builder report;
  report.table("") = p::to_report_table(table, {.display = {{"x", decimal}}});
  EXPECT_NE(p::render_plain(report).find("0.333333"), std::string::npos);
  EXPECT_FALSE(p::data_column<mpreal>("x").fractional().fixed(2).display().fractions);
}

TEST(MprealDataTable, ApproximateDelimitedOutputRoundTripsAndDisplayIsOptIn)
{
  for (auto bits : {128, 256})
  {
    auto precision = Precision::bits(bits);
    auto value = mpreal{1}.at(precision) + numeric_limits<mpreal>::epsilon(precision);
    auto table = p::make_data_table("precise", p::data_column<mpreal>("x").fixed(2));
    table.append(value);
    std::ostringstream csv, rounded;
    p::write_csv(csv, table);
    auto text = csv.str().substr(2);
    text.pop_back();
    EXPECT_EQ(parse_real<mpreal>(text, precision), value);
    p::write_csv(rounded, table, {.precision = p::data_export_precision::display});
    EXPECT_EQ(rounded.str(), "x\n1.00\n");
  }
}

TEST(MprealDataTable, JsonSnapshotsStreamingAndSpecialValues)
{
  auto table = p::make_data_table("mixed", p::data_column<std::optional<mpreal>>("x").fixed(1));
  table.append(std::nullopt);
  table.append(mpreal("1/3", Precision::exact()));
  std::ostringstream live;
  table.attach(p::json_sink(live));
  for (auto bits : {128, 256})
    for (auto text : {"1.000000000000000000000000000000000003", "1e10000", "-0", "inf", "-inf", "nan"})
      table.append(mpreal(text, Precision::bits(bits)));
  table.finish();
  std::ostringstream snapshot, replay;
  p::write_json(snapshot, table);
  table.attach(p::json_sink(replay));
  EXPECT_EQ(live.str(), snapshot.str());
  EXPECT_EQ(replay.str(), snapshot.str());
  EXPECT_NE(snapshot.str().find("\"precision\":\"per_value\",\"encoding\":\"value_precision\""), std::string::npos);
  EXPECT_NE(snapshot.str().find("\"rows\":[[null],[{\"value\":\"1/3\",\"precision\":\"exact\"}]"), std::string::npos);
  for (auto bits : {128, 256})
    for (auto text : {"-0", "inf", "-inf", "nan"})
      EXPECT_NE(snapshot.str().find(fmt::format("{{\"value\":\"{}\",\"precision\":{}}}", text, bits)),
                std::string::npos);
}

TEST(MprealDataTable, JsonNumericalValuesReconstructAtRecordedPrecision)
{
  for (auto bits : {128, 256})
    for (auto input : {"1.000000000000000000000000000000000003", "1e10000", "1e-10000"})
    {
      auto precision = Precision::bits(bits);
      auto value = mpreal(input, precision);
      auto table = p::make_data_table("precise", p::data_column<mpreal>("x").fixed(1));
      table.append(value);
      std::ostringstream output;
      p::write_json(output, table);
      auto json = output.str();
      std::string prefix = "\"rows\":[[{\"value\":\"";
      auto start = json.find(prefix);
      ASSERT_NE(start, std::string::npos);
      start += prefix.size();
      auto end = json.find('"', start);
      ASSERT_NE(end, std::string::npos);
      EXPECT_EQ(parse_real<mpreal>(json.substr(start, end - start), precision), value);
      EXPECT_NE(json.find(fmt::format("\"precision\":{}", bits), end), std::string::npos);
    }
}

TEST(MprealDataTable, ExportUsesDecimalPointAndRestoresThreadLocale)
{
  struct locale_scope
  {
      locale_t locale = newlocale(LC_NUMERIC_MASK, "en_DK.utf8", nullptr);
      locale_t previous = nullptr;
      ~locale_scope()
      {
        if (previous) uselocale(previous);
        if (locale) freelocale(locale);
      }
  } scope;
  if (!scope.locale) GTEST_SKIP() << "en_DK.utf8 numeric locale is unavailable";
  scope.previous = uselocale(scope.locale);
  ASSERT_NE(scope.previous, nullptr);
  ASSERT_STREQ(localeconv()->decimal_point, ",");
  auto table = p::make_data_table("decimal", p::data_column<mpreal>("x").fixed(2));
  table.append(mpreal("1.25", Precision::bits(128)));
  table.append(mpreal("5/4", Precision::exact()));
  std::ostringstream output;
  p::write_csv(output, table);
  EXPECT_EQ(output.str(), "x\n1.25\n1.25\n");
  EXPECT_EQ(uselocale(nullptr), scope.locale);
  EXPECT_STREQ(localeconv()->decimal_point, ",");
}

TEST(MprealMetadata, RetainsAndConvertsOnlyWithExplicitPrecision)
{
  auto precision = Precision::bits(256);
  auto x = mpreal{1}.at(precision) + numeric_limits<mpreal>::epsilon(precision);
  metadata_value stored(x);
  EXPECT_EQ(stored.get<mpreal>(), x);
  EXPECT_EQ(stored.as<mpreal>().precision(), precision);
  EXPECT_EQ(stored.as<mpreal>(Precision::bits(128)).precision(), Precision::bits(128));
  EXPECT_EQ(stored.as<double>(), 1.0);
  EXPECT_THROW((void)stored.as<mpreal>(Precision::exact()), std::invalid_argument);
  EXPECT_THROW((void)metadata_value("1/3").as<mpreal>(), std::invalid_argument);
  EXPECT_THROW((void)metadata_value(1.0).as<mpreal>(), std::invalid_argument);
  EXPECT_EQ(metadata_value("1/3").as<mpreal>(precision), mpreal("1/3", precision));
  EXPECT_TRUE(metadata_value("1/3").as<mpreal>(Precision::exact()).is_exact());
  EXPECT_EQ(metadata_value(1.25L).as<mpreal>(precision), mpreal("5/4", precision));
  EXPECT_TRUE(metadata_value(7).as<mpreal>(Precision::exact()).is_exact());
  EXPECT_THROW((void)metadata_value(mpreal("1e10000", precision)).as<double>(), std::out_of_range);
  EXPECT_THROW((void)metadata_value(mpreal(uninitialized)), std::invalid_argument);
  EXPECT_THROW((void)metadata_value(std::optional<mpreal>(mpreal(uninitialized))), std::invalid_argument);
  metadata_value missing(std::optional<mpreal>{});
  EXPECT_FALSE(missing.as<std::optional<mpreal>>(precision));
  EXPECT_THROW((void)missing.as<mpreal>(precision), std::invalid_argument);
  EXPECT_EQ(metadata_value(std::optional<mpreal>(x)).as<mpreal>().precision(), precision);
}

TEST(MprealMetadata, OutputSessionUsesExactFormattingWithoutMutatingMetadata)
{
  metadata_document metadata;
  metadata.group("parameters");
  metadata.add("parameters", "ratio", mpreal("1/3", Precision::exact()),
               {.display = {.numeric = {.precision = 6, .notation = real_format_notation::fixed}}});
  EXPECT_EQ(metadata.strings().at("ratio"), "0.333333");
  EXPECT_TRUE(metadata.find("ratio")->value.get<mpreal>().is_exact());
  auto output = std::make_shared<std::ostringstream>();
  output_session session(metadata);
  session.stream("csv", output, output_format::commented_csv);
  auto table = p::make_data_table("values", p::data_column<mpreal>("x").fractional());
  session.attach(table, "values");
  table.append(mpreal("2/7", Precision::exact()));
  table.finish();
  session.finish();
  EXPECT_NE(output->str().find("# ratio: 0.333333\n"), std::string::npos);
  EXPECT_NE(output->str().find("x\n2/7\n"), std::string::npos);
}
} // namespace
#endif
