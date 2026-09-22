#include <gtest/gtest.h>
#include <uni20/common/data_table_sinks.hpp>

#include <sstream>

namespace
{
namespace p = uni20::presentation;

TEST(DataTableJson, EncodesExactTypedValuesAndMissingness)
{
  auto table =
      p::make_data_table("β\"\n", p::data_column<unsigned>("id"), p::data_column<std::uint64_t>("large"),
                         p::data_column<uni20::half_int>("spin").fractional(), p::data_column<double>("x").fixed(1),
                         p::data_column<std::optional<double>>("missing"), p::data_column<bool>("ok"),
                         p::data_column<std::string>("text"));
  table.append(1, uni20::numeric_limits<std::uint64_t>::max(), uni20::from_twice(std::int64_t{9007199254740993}),
               1.2345678901234567, std::nullopt, true, std::string("a\0b\n\"\\", 7));
  std::ostringstream out;
  out.width(100);
  out.precision(1);
  out << std::hex << std::showbase;
  p::write_json(out, table);
  EXPECT_TRUE(out.str().starts_with("{\"title\":\"β\\\"\\u000a\""));
  EXPECT_NE(out.str().find("\"type\":\"uint64\",\"encoding\":\"decimal_string\""), std::string::npos);
  EXPECT_NE(out.str().find("\"type\":\"half_int\",\"storage_bits\":64,\"encoding\":\"twice_decimal_string\""),
            std::string::npos);
  EXPECT_NE(out.str().find("\"type\":\"real\",\"radix\":2,\"precision_bits\":53,\"encoding\":\"decimal_string\""),
            std::string::npos);
  EXPECT_NE(out.str().find("\"rows\":[[1,\"18446744073709551615\",\"9007199254740993\",\"1.2345678901234567\",null,"
                           "true,\"a\\u0000b\\u000a\\\"\\\\\\u0000\"]]"),
            std::string::npos);
  EXPECT_EQ(out.str().find("\"summary\""), std::string::npos);
  EXPECT_EQ(out.width(), 100);
}

TEST(DataTableJson, NonfiniteValuesAreStringsAndSignedZeroIsPreserved)
{
  auto table = p::make_data_table("special", p::data_column<std::optional<double>>("x"));
  table.append(std::nullopt);
  table.append(-0.0);
  table.append(uni20::numeric_limits<double>::infinity());
  table.append(-uni20::numeric_limits<double>::infinity());
  table.append(uni20::numeric_limits<double>::quiet_NaN());
  std::ostringstream out;
  p::write_json(out, table);
  EXPECT_NE(out.str().find("\"rows\":[[null],[\"-0\"],[\"inf\"],[\"-inf\"],[\"nan\"]]"), std::string::npos);
}

TEST(DataTableJson, MetadataSummaryAndProjectionSurviveSnapshotAndStreaming)
{
  auto table = p::make_data_table("run", {.metadata = {{"U", "4.0"}, {"model", "Hubbard"}}}, p::data_column<int>("id"),
                                  p::data_column<long double>("energy").unit("t").description("Total energy"));
  table.append(1, -6.25L);
  std::ostringstream live, snapshot;
  table.attach(p::json_sink(live, {.columns = {"energy", "id"}}));
  table.append(2, -6.125L);
  table.finish({{"cpu_seconds", "1.125"}, {"status", "ok"}});
  p::write_json(snapshot, table, {.columns = {"energy", "id"}});
  EXPECT_EQ(live.str(), snapshot.str());
  EXPECT_NE(live.str().find("\"metadata\":{\"U\":\"4.0\",\"model\":\"Hubbard\"}"), std::string::npos);
  EXPECT_NE(live.str().find("\"rows\":[[\"-6.25\",1],[\"-6.125\",2]]"), std::string::npos);
  EXPECT_NE(live.str().find("\"summary\":{\"cpu_seconds\":\"1.125\",\"status\":\"ok\"}"), std::string::npos);
  std::ostringstream late;
  table.attach(p::json_sink(late, {.columns = {"energy", "id"}}));
  EXPECT_EQ(late.str(), snapshot.str());
}

TEST(DataTableJson, InvalidUtf8IsRejectedRatherThanWritingInvalidJson)
{
  for (std::string invalid : {std::string("\xc0\xaf"), std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80"),
                              std::string("\xe2"), std::string("\xe2\x20\x80"), std::string("\xe0\x80\x80")})
  {
    auto table = p::make_data_table("text", p::data_column<std::string>("x"));
    table.append(invalid);
    std::ostringstream out;
    EXPECT_THROW(p::write_json(out, table), std::invalid_argument);
  }
  auto valid = p::make_data_table("Unicode", p::data_column<std::string>("x"));
  valid.append("π 中 🎲");
  std::ostringstream out;
  EXPECT_NO_THROW(p::write_json(out, valid));
  EXPECT_NE(out.str().find("π 中 🎲"), std::string::npos);
}

TEST(DataTableJson, EmptyRowsAndDestructionDoNotInventFinalSummary)
{
  auto empty = p::make_data_table("empty", p::data_column<int>("x"));
  std::ostringstream snapshot;
  p::write_json(snapshot, empty);
  EXPECT_TRUE(snapshot.str().ends_with("\"rows\":[]}\n"));
  std::ostringstream stream;
  {
    auto table = p::make_data_table("unfinished", p::data_column<int>("x"));
    table.attach(p::json_sink(stream));
    table.append(1);
  }
  EXPECT_TRUE(stream.str().ends_with("\"rows\":[[1]"));
}

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK && defined(MPLAPACK_BINARY128_MODE) &&                      \
    (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_FLOAT128)
TEST(DataTableJson, NativeBinary128JsonRetainsDigitsBeyondLongDouble)
{
  using Real = uni20::float128;
  auto table = p::make_data_table("quad", p::data_column<Real>("x").fixed(1));
  Real value = uni20::parse_real<Real>("1.0000000000000000000000000000000002");
  table.append(value);
  std::ostringstream out;
  p::write_json(out, table);
  EXPECT_NE(out.str().find("\"precision_bits\":113"), std::string::npos);
  auto start = out.str().find("\"rows\":[[\"") + 10;
  auto end = out.str().find('"', start);
  EXPECT_EQ(uni20::parse_real<Real>(out.str().substr(start, end - start)), value);
  EXPECT_NE(static_cast<Real>(static_cast<long double>(value)), value);
  std::string rendered;
  uni20::display::scoped_sink capture([&](uni20::display::event const& event) {
    rendered += std::visit([](auto const& content) { return p::render_plain(content); }, event.content);
  });
  table.attach(p::terminal_sink({.wrap_width = 60}));
  table.finish();
  EXPECT_NE(rendered.find("1.0"), std::string::npos);
}
#endif
} // namespace
