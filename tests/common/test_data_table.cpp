#include <gtest/gtest.h>
#include <uni20/common/data_table.hpp>

#include <cmath>
#include <locale.h>
#include <locale>
#include <sstream>
#include <streambuf>
#include <type_traits>

namespace
{
namespace p = uni20::presentation;

template <typename Table, typename... Args>
concept CanAppend = requires(Table& table, Args&&... args) { table.append(std::forward<Args>(args)...); };

using Counts = p::data_table<unsigned, bool>;
static_assert(CanAppend<Counts, int, bool>);
static_assert(!CanAppend<Counts, double, bool>);
static_assert(!CanAppend<Counts, int, int>);
static_assert(!CanAppend<Counts, int>);
static_assert(!CanAppend<Counts, int, bool, int>);
static_assert(!p::DataTableValue<std::string_view>);
static_assert(!p::DataTableValue<std::optional<std::string>>);
static_assert(!p::DataTableValue<uni20::complex<double>>);
static_assert(!CanAppend<p::data_table<uni20::half_int>, double>);
static_assert(CanAppend<p::data_table<std::optional<uni20::half_int>>, std::nullopt_t>);

TEST(DataTable, OwnsValuesAndUsesSchemaConversions)
{
  auto table = p::make_data_table("results", p::data_column<unsigned>("index"), p::data_column<long double>("energy"),
                                  p::data_column<std::optional<double>>("gap"), p::data_column<std::string>("label"),
                                  p::data_column<bool>("ok"));
  std::string source = "original";
  table.append(1, -6.25, 0, std::string_view(source), true);
  table.append(2, 0, std::nullopt, std::string("temporary"), false);
  table.append(3, 0, std::optional<int>{4}, "literal", true);
  table.append(4, 0, std::optional<int>{}, "empty gap", true);
  source[0] = 'X';
  ASSERT_EQ(table.size(), 4U);
  EXPECT_EQ(std::get<3>(table.rows()[0]), "original");
  EXPECT_EQ(std::get<1>(table.rows()[0]), -6.25L);
  EXPECT_EQ(std::get<2>(table.rows()[0]), 0.0);
  EXPECT_FALSE(std::get<2>(table.rows()[1]));
  EXPECT_EQ(std::get<2>(table.rows()[2]), 4.0);
  EXPECT_FALSE(std::get<2>(table.rows()[3]));
  EXPECT_EQ(std::get<3>(table.rows()[1]), "temporary");
  static_assert(std::same_as<decltype(std::get<1>(table.rows()[0])), long double const&>);
}

TEST(DataTable, CheckedRowConstructionIsIndependentOfStorage)
{
  using Table = p::data_table<std::uint8_t, std::optional<std::int8_t>>;
  auto row = Table::make_row(255, -128);
  EXPECT_EQ(std::get<0>(row), 255);
  EXPECT_EQ(std::get<1>(row), -128);
  EXPECT_THROW((void)Table::make_row(256, 0), std::overflow_error);
  EXPECT_THROW((void)Table::make_row(0, std::optional<int>{128}), std::overflow_error);
}

TEST(DataTable, RejectsIntegerOverflowBeforeStoringAnyPartOfARow)
{
  auto table = p::make_data_table("counts", p::data_column<std::string>("name"),
                                  p::data_column<std::uint64_t>("unsigned"), p::data_column<std::int64_t>("signed"));
  auto const umax = uni20::numeric_limits<std::uint64_t>::max();
  auto const imin = uni20::numeric_limits<std::int64_t>::min();
  auto const imax = uni20::numeric_limits<std::int64_t>::max();
  table.append("limits", umax, imin);
  table.append("positive", 0, imax);
  EXPECT_THROW(table.append("negative unsigned", -1, 0), std::overflow_error);
  EXPECT_THROW(table.append("unsigned too large", 0, umax), std::overflow_error);
  EXPECT_EQ(table.size(), 2U);
  EXPECT_EQ(std::get<1>(table.rows()[0]), umax);
  EXPECT_EQ(std::get<2>(table.rows()[0]), imin);
}

TEST(DataTable, ValidatesSchemaAndOwnsMetadata)
{
  EXPECT_THROW(p::data_column<int>(""), std::invalid_argument);
  EXPECT_THROW(p::data_column<int>("1energy"), std::invalid_argument);
  EXPECT_THROW(p::data_column<int>("energy\tgap"), std::invalid_argument);
  EXPECT_THROW((void)p::make_data_table("bad", p::data_column<int>("same"), p::data_column<int>("same")),
               std::invalid_argument);
  EXPECT_THROW(p::data_column<double>("value").fixed(-1), std::invalid_argument);
  EXPECT_THROW(p::data_column<double>("value").general(0), std::invalid_argument);
  std::string label = "Energy";
  auto table =
      p::make_data_table("result", p::data_column<double>("energy").label(label).unit("J").description("Total energy"));
  label.clear();
  auto const& col = std::get<0>(table.columns());
  EXPECT_EQ(col.label(), "Energy");
  EXPECT_EQ(col.unit(), "J");
  EXPECT_EQ(col.description(), "Total energy");
}

TEST(DataTable, HalfIntegersRemainExactAndCheckWholeIntegerInsertion)
{
  auto table = p::make_data_table("spin", p::data_column<uni20::half_int>("spin").fractional(),
                                  p::data_column<std::optional<uni20::basic_half_int<std::int8_t>>>("small"));
  table.append(uni20::from_twice(3), uni20::from_twice(-1));
  table.append(uni20::from_twice(std::int64_t{9007199254740993}), 63);
  table.append(uni20::from_twice(uni20::numeric_limits<std::int64_t>::min()), -64);
  table.append(uni20::from_twice(uni20::numeric_limits<std::int64_t>::max()), std::nullopt);
  EXPECT_THROW(table.append(0, 64), std::overflow_error);
  EXPECT_THROW(table.append(0, -65), std::overflow_error);
  EXPECT_THROW(table.append(0, uni20::from_twice(128)), std::overflow_error);
  EXPECT_THROW(table.append(uni20::numeric_limits<std::uint64_t>::max(), 0), std::overflow_error);
  EXPECT_THROW(table.append(0, std::uint64_t{256}), std::overflow_error);
  EXPECT_EQ(table.size(), 4U);
  std::ostringstream csv;
  p::write_csv(csv, table);
  EXPECT_EQ(csv.str(),
            "spin,small\n1.5,-0.5\n4503599627370496.5,63\n-4611686018427387904,-64\n4611686018427387903.5,\n");
  auto report = p::to_report_table(table);
  auto const& row = std::get<std::vector<p::table_cell>>(report.entries()[0]);
  EXPECT_EQ(p::render_plain(row[0].content), "3/2");
  EXPECT_EQ(p::render_plain(row[1].content), "-0.5");
}

TEST(DataTable, DisplayRoundingDoesNotChangeStoredOrExportedValues)
{
  auto table = p::make_data_table("values", p::data_column<double>("energy").label("Energy").fixed(2),
                                  p::data_column<std::optional<double>>("missing").missing("unavailable"));
  table.append(1.2345678901234567, std::nullopt);
  auto report = p::to_report_table(table);
  auto const& row = std::get<std::vector<p::table_cell>>(report.entries()[0]);
  EXPECT_EQ(p::render_plain(row[0].content), "1.23");
  EXPECT_EQ(p::render_plain(row[1].content), "unavailable");
  EXPECT_EQ(report.columns()[0].heading, "Energy");
  std::ostringstream csv, rounded;
  p::write_csv(csv, table);
  p::write_csv(rounded, table, {.precision = p::data_export_precision::display});
  EXPECT_EQ(csv.str(), "energy,missing\n1.2345678901234567,\n");
  EXPECT_EQ(rounded.str(), "energy,missing\n1.23,\n");
  EXPECT_EQ(std::get<0>(table.rows()[0]), 1.2345678901234567);
}

TEST(DataTable, LongDoubleRoundTripsWithoutPassingThroughDouble)
{
  long double const value = 1.0L + uni20::numeric_limits<long double>::epsilon();
  auto table = p::make_data_table("precision", p::data_column<long double>("x").fixed(2));
  table.append(value);
  std::ostringstream output;
  p::write_tsv(output, table);
  std::istringstream input(output.str());
  input.imbue(std::locale::classic());
  std::string header;
  long double result = 0;
  input >> header >> result;
  EXPECT_EQ(result, value);
  if constexpr (uni20::numeric_limits<long double>::digits > uni20::numeric_limits<double>::digits)
  {
    EXPECT_NE(result, static_cast<long double>(static_cast<double>(value)));
  }
}

TEST(DataTable, MissingNonfiniteAndNegativeZeroAreDistinct)
{
  auto table = p::make_data_table("exceptional", p::data_column<std::optional<double>>("x"));
  table.append(std::nullopt);
  table.append(-0.0);
  table.append(uni20::numeric_limits<double>::quiet_NaN());
  table.append(uni20::numeric_limits<double>::infinity());
  table.append(-uni20::numeric_limits<double>::infinity());
  std::ostringstream csv;
  p::write_csv(csv, table);
  EXPECT_EQ(csv.str(), "x\n\"\"\n-0\nnan\ninf\n-inf\n");
}

TEST(DataTable, QuotingPreservesStringsAndEmptyFields)
{
  auto table = p::make_data_table("text", p::data_column<std::string>("text"),
                                  p::data_column<std::optional<int>>("count"), p::data_column<bool>("ok"));
  table.append("comma, quote\" tab\t newline\nreturn\r", std::nullopt, true);
  table.append("", 2, false);
  std::ostringstream csv, tsv;
  p::write_csv(csv, table);
  p::write_tsv(tsv, table);
  EXPECT_EQ(csv.str(), "text,count,ok\n\"comma, quote\"\" tab\t newline\nreturn\r\",,true\n\"\",2,false\n");
  EXPECT_EQ(tsv.str(), "text\tcount\tok\n\"comma, quote\"\" tab\t newline\nreturn\r\"\t\ttrue\n\"\"\t2\tfalse\n");
  char const* null_text = nullptr;
  EXPECT_THROW(table.append(null_text, 1, true), std::invalid_argument);
  EXPECT_EQ(table.size(), 2U);
}

struct comma_punctuation : std::numpunct<char>
{
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
};

TEST(DataTable, ExportIgnoresStreamLocalePrecisionAndWidth)
{
  auto table = p::make_data_table("locale", p::data_column<double>("x"), p::data_column<int>("n"));
  table.append(1234.5, 1234);
  std::ostringstream output;
  output.imbue(std::locale(std::locale::classic(), new comma_punctuation));
  output.precision(1);
  output.width(50);
  output << std::fixed;
  p::write_tsv(output, table);
  EXPECT_EQ(output.str(), "x\tn\n1234.5\t1234\n");
}

struct failing_buffer : std::streambuf
{
    int_type overflow(int_type) override { return traits_type::eof(); }
};

TEST(DataTable, ReportsWriteFailureWithoutStreamExceptionsAndDoesNotModifyRows)
{
  auto table = p::make_data_table("failure", p::data_column<int>("x"));
  table.append(3);
  failing_buffer buffer;
  std::ostream output(&buffer);
  EXPECT_THROW(p::write_csv(output, table), std::ios_base::failure);
  EXPECT_EQ(table.size(), 1U);
  EXPECT_EQ(std::get<0>(table.rows()[0]), 3);
  std::ostringstream snapshot;
  p::write_csv(snapshot, table);
  EXPECT_EQ(snapshot.str(), "x\n3\n");
}

TEST(DataTable, EmptyTableExportsOnlyHeaders)
{
  auto table = p::make_data_table("empty", p::data_column<int>("x"), p::data_column<double>("y"));
  std::ostringstream csv;
  p::write_csv(csv, table);
  EXPECT_EQ(csv.str(), "x,y\n");
  EXPECT_TRUE(p::to_report_table(table).entries().empty());
}

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK && defined(MPLAPACK_BINARY128_MODE) &&                      \
    (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_FLOAT128)
TEST(DataTable, Binary128RetainsPrecisionAndFormatsDisplaySeparately)
{
  using Real = uni20::float128;
  Real const value = uni20::parse_real<Real>("1.0000000000000000000000000000000002");
  auto table = p::make_data_table("quad", p::data_column<Real>("x").fixed(2));
  table.append(value);
  std::ostringstream output;
  p::write_csv(output, table);
  auto text = output.str();
  EXPECT_EQ(uni20::parse_real<Real>(text.substr(2, text.size() - 3)), value);
  EXPECT_NE(static_cast<Real>(static_cast<long double>(value)), value);
  auto report = p::to_report_table(table);
  EXPECT_EQ(p::render_plain(std::get<std::vector<p::table_cell>>(report.entries()[0])[0].content), "1.00");
}

TEST(DataTable, Binary128ExportUsesDecimalPointAndRestoresThreadLocale)
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
  auto table = p::make_data_table("quad", p::data_column<uni20::float128>("x").fixed(2));
  table.append(1.25);
  std::ostringstream output;
  p::write_csv(output, table);
  EXPECT_EQ(output.str(), "x\n1.25\n");
  auto report = p::to_report_table(table);
  EXPECT_EQ(p::render_plain(std::get<std::vector<p::table_cell>>(report.entries()[0])[0].content), "1.25");
  EXPECT_EQ(uselocale(nullptr), scope.locale);
  EXPECT_STREQ(localeconv()->decimal_point, ",");
}
#endif
} // namespace
