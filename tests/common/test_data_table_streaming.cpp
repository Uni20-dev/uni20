#include <gtest/gtest.h>
#include <uni20/common/data_table_sinks.hpp>

#include <functional>
#include <sstream>

namespace
{
namespace p = uni20::presentation;

struct recording_sink
{
    std::string name;
    std::vector<std::string>* events;
    std::optional<int> fail_row = std::nullopt;
    bool fail_begin = false;
    bool fail_finish = false;
    std::function<void()> callback = {};

    template <typename Schema>
    void begin(std::string const&, Schema const&, p::table_metadata const& metadata, p::data_sink_start start)
    {
      events->push_back(name + ":begin:" + std::to_string(start.first_row));
      if (metadata.contains("model")) events->push_back(name + ":model:" + metadata.at("model"));
      if (fail_begin) throw std::runtime_error("begin failed");
    }
    template <typename Schema, typename Row> void row(Schema const&, Row const& row)
    {
      int value = std::get<0>(row);
      events->push_back(name + ":row:" + std::to_string(value));
      if (callback) callback();
      if (fail_row == value) throw std::runtime_error("row failed");
    }
    void finish(p::table_metadata const& summary)
    {
      events->push_back(name + ":finish");
      if (summary.contains("status")) events->push_back(name + ":status:" + summary.at("status"));
      if (fail_finish) throw std::runtime_error("finish failed");
    }
};

TEST(DataTableStreaming, LateAttachmentReplaysThenFollowsInOrder)
{
  auto table = p::make_data_table("run", p::data_column<int>("x"));
  std::vector<std::string> events;
  table.append(1);
  auto a = table.attach(recording_sink{"a", &events});
  table.append(2);
  auto b = table.attach(recording_sink{"b", &events});
  table.append(3);
  table.finish();
  EXPECT_EQ(events, (std::vector<std::string>{"a:begin:0", "a:row:1", "a:row:2", "b:begin:0", "b:row:1", "b:row:2",
                                              "a:row:3", "b:row:3", "a:finish", "b:finish"}));
  EXPECT_EQ(table.sink_state(a.id), p::data_sink_state::closed);
  EXPECT_EQ(table.sink_state(b.id), p::data_sink_state::closed);
  EXPECT_EQ(table.size(), 3U);
}

TEST(DataTableStreaming, RequiredAndOptionalFailuresDoNotPreventLaterDelivery)
{
  auto table = p::make_data_table("run", p::data_column<int>("x"));
  std::vector<std::string> events;
  auto a = table.attach(recording_sink{"a", &events, 2});
  auto b = table.attach(recording_sink{"b", &events, 2}, {.required = false});
  table.attach(recording_sink{"c", &events});
  table.append(1);
  try
  {
    table.append(2);
    FAIL() << "required failure must throw";
  }
  catch (p::data_delivery_error const& error)
  {
    ASSERT_EQ(error.report().failures.size(), 2U);
    EXPECT_EQ(error.report().failures[0].sink, a.id);
    EXPECT_TRUE(error.report().failures[0].required);
    EXPECT_EQ(error.report().failures[1].sink, b.id);
    EXPECT_FALSE(error.report().failures[1].required);
    EXPECT_THROW(std::rethrow_exception(error.report().failures[0].exception), std::runtime_error);
  }
  EXPECT_EQ(events.back(), "c:row:2");
  EXPECT_EQ(table.size(), 2U);
  EXPECT_EQ(std::get<0>(table.rows().back()), 2);
  EXPECT_EQ(table.sink_state(a.id), p::data_sink_state::failed);
  auto count = events.size();
  table.append(3);
  EXPECT_EQ(events.size(), count + 1);
  EXPECT_EQ(events.back(), "c:row:3");
}

TEST(DataTableStreaming, OptionalFailureIsReturnedAndDisabled)
{
  auto table = p::make_data_table("run", p::data_column<int>("x"));
  std::vector<std::string> events;
  auto id = table.attach(recording_sink{"screen", &events, 1}, {.required = false}).id;
  auto report = table.append(1);
  ASSERT_EQ(report.failures.size(), 1U);
  EXPECT_EQ(report.failures[0].sink, id);
  EXPECT_EQ(report.failures[0].operation, p::data_sink_operation::row);
  EXPECT_FALSE(report.failures[0].required);
  EXPECT_TRUE(table.append(2).failures.empty());
  EXPECT_EQ(events.size(), 2U);
}

TEST(DataTableStreaming, InvalidRowsNeverReachStorageOrSinks)
{
  auto table = p::make_data_table("run", p::data_column<std::int8_t>("x"));
  std::vector<std::string> events;
  table.attach(recording_sink{"a", &events});
  EXPECT_THROW(table.append(128), std::overflow_error);
  EXPECT_EQ(table.size(), 0U);
  EXPECT_EQ(events, (std::vector<std::string>{"a:begin:0"}));
}

TEST(DataTableStreaming, FailedBeginAndReplayNeverSubscribeOrRetry)
{
  auto table = p::make_data_table("run", p::data_column<int>("x"));
  table.append(1);
  table.append(2);
  table.append(3);
  std::vector<std::string> events;
  auto begin = table.attach(recording_sink{"begin", &events, {}, true}, {.required = false});
  auto replay = table.attach(recording_sink{"replay", &events, 2}, {.required = false});
  ASSERT_EQ(begin.report.failures.size(), 1U);
  ASSERT_EQ(replay.report.failures.size(), 1U);
  EXPECT_EQ(begin.report.failures[0].operation, p::data_sink_operation::begin);
  EXPECT_EQ(replay.report.failures[0].operation, p::data_sink_operation::row);
  EXPECT_EQ(table.sink_state(replay.id), p::data_sink_state::failed);
  table.append(4);
  table.finish();
  EXPECT_EQ(events, (std::vector<std::string>{"begin:begin:0", "replay:begin:0", "replay:row:1", "replay:row:2"}));
}

TEST(DataTableStreaming, IndividualFinishAndFinishedAttachmentPreserveSummary)
{
  auto table = p::make_data_table("run", {.metadata = {{"model", "Hubbard"}}}, p::data_column<int>("x"));
  std::vector<std::string> events;
  auto a = table.attach(recording_sink{"a", &events});
  table.finish_sink(a.id);
  table.append(1);
  table.finish({{"status", "ok"}});
  auto b = table.attach(recording_sink{"b", &events});
  auto count = events.size();
  table.finish();
  table.finish({{"status", "ok"}});
  table.finish_sink(b.id);
  EXPECT_EQ(events.size(), count);
  EXPECT_EQ(events, (std::vector<std::string>{"a:begin:0", "a:model:Hubbard", "a:finish", "b:begin:0",
                                              "b:model:Hubbard", "b:row:1", "b:finish", "b:status:ok"}));
  EXPECT_THROW(table.append(2), std::logic_error);
  EXPECT_THROW(table.finish({{"status", "changed"}}), std::logic_error);
  EXPECT_EQ(table.retained_size(), 1U);
}

TEST(DataTableStreaming, FinishAttemptsAllOutputsAndDoesNotRetry)
{
  auto table = p::make_data_table("run", p::data_column<int>("x"));
  std::vector<std::string> events;
  table.attach(recording_sink{"a", &events, {}, false, true});
  table.attach(recording_sink{"b", &events, {}, false, true}, {.required = false});
  table.attach(recording_sink{"c", &events});
  try
  {
    table.finish();
    FAIL();
  }
  catch (p::data_delivery_error const& error)
  {
    EXPECT_EQ(error.report().failures.size(), 2U);
  }
  auto count = events.size();
  EXPECT_THROW(table.finish(), p::data_delivery_error);
  EXPECT_EQ(events.size(), count);
  EXPECT_EQ(events.back(), "c:finish");
  EXPECT_TRUE(table.finished());
}

struct counting_sink
{
    std::size_t* count;
    template <typename Schema>
    void begin(std::string const&, Schema const&, p::table_metadata const&, p::data_sink_start)
    {}
    template <typename Schema, typename Row> void row(Schema const&, Row const&) { ++*count; }
    void finish(p::table_metadata const&) {}
};

TEST(DataTableStreaming, NoRetentionHasNoHistoryAndLateAttachmentMustBeExplicit)
{
  auto table = p::make_data_table("stream", {.retain = p::retention::none}, p::data_column<int>("x"));
  std::size_t count = 0;
  table.attach(counting_sink{&count});
  for (int i = 0; i < 1000; ++i)
    table.append(i);
  EXPECT_EQ(count, 1000U);
  EXPECT_EQ(table.size(), 1000U);
  EXPECT_EQ(table.retained_size(), 0U);
  std::ostringstream out;
  EXPECT_THROW(p::write_csv(out, table), std::logic_error);
  EXPECT_THROW(p::write_tsv(out, table), std::logic_error);
  EXPECT_THROW(p::write_json(out, table), std::logic_error);
  EXPECT_THROW((void)p::to_report_table(table), std::logic_error);
  EXPECT_TRUE(out.str().empty());
  EXPECT_THROW(table.attach(p::json_sink(out)), std::logic_error);
  EXPECT_TRUE(out.str().empty());
  table.attach(p::json_sink(out), {.replay = p::sink_replay::future_only});
  table.append(1000);
  table.finish();
  EXPECT_NE(out.str().find("\"first_row\":1000"), std::string::npos);
  EXPECT_NE(out.str().find("\"rows\":[[1000]]"), std::string::npos);
}

TEST(DataTableStreaming, ProjectionsAndFormattingAreIndependentAcrossSinks)
{
  auto table = p::make_data_table("run", p::data_column<int>("id"), p::data_column<double>("energy").fixed(3));
  table.append(1, 1.2345678901234567);
  std::ostringstream full, rounded;
  table.attach(p::csv_sink(full));
  table.attach(p::tsv_sink(
      rounded,
      {.precision = p::data_export_precision::display,
       .projection = {
           .columns = {"energy"},
           .display = {{"energy", {.numeric = {.precision = 1, .notation = uni20::real_format_notation::fixed}}}}}}));
  table.append(2, 2.5);
  table.finish();
  EXPECT_EQ(full.str(), "id,energy\n1,1.2345678901234567\n2,2.5\n");
  EXPECT_EQ(rounded.str(), "energy\n1.2\n2.5\n");
  EXPECT_EQ(std::get<1>(table.rows()[0]), 1.2345678901234567);
  EXPECT_EQ(std::get<1>(table.columns()).display().numeric.precision, 3);
  std::ostringstream invalid;
  EXPECT_THROW(p::write_csv(invalid, table, {.projection = {.columns = {"absent"}}}), std::invalid_argument);
  EXPECT_THROW(p::write_csv(invalid, table, {.projection = {.columns = {"id", "id"}}}), std::invalid_argument);
  EXPECT_TRUE(invalid.str().empty());
}

TEST(DataTableStreaming, TerminalAdapterReplaysFormattedCellsThroughDisplayRouter)
{
  std::vector<std::string> rendered;
  uni20::display::scoped_sink capture([&](uni20::display::event const& event) {
    rendered.push_back(std::visit([](auto const& content) { return p::render_plain(content); }, event.content));
  });
  auto table =
      p::make_data_table("run", p::data_column<int>("id"), p::data_column<uni20::half_int>("spin").fractional());
  table.append(1, uni20::from_twice(3));
  table.attach(p::terminal_sink({.projection = {.columns = {"spin"}}, .wrap_width = 60}));
  table.append(2, uni20::from_twice(-1));
  table.finish();
  std::string text;
  for (auto const& line : rendered)
    text += line;
  EXPECT_NE(text.find("3/2"), std::string::npos);
  EXPECT_NE(text.find("-1/2"), std::string::npos);
  EXPECT_EQ(text.find("id"), std::string::npos);
}

TEST(DataTableStreaming, TerminalAdapterPreservesDecimalAlignmentAfterFormatting)
{
  std::vector<std::string> rendered;
  uni20::display::scoped_sink capture([&](uni20::display::event const& event) {
    rendered.push_back(std::visit([](auto const& content) { return p::render_plain(content); }, event.content));
  });
  auto table = p::make_data_table("", p::data_column<double>("x"));
  table.attach(p::terminal_sink({.wrap_width = 60}));
  table.append(12.5);
  table.append(1.25);
  table.finish();
  ASSERT_EQ(rendered.size(), 2U);
  auto first_row_start = rendered[0].rfind('\n', rendered[0].size() - 2) + 1;
  EXPECT_EQ(rendered[0].find('.', first_row_start) - first_row_start, rendered[1].find('.'));
}

TEST(DataTableStreaming, ReentrantMutationIsRejectedWithoutAcceptingAnotherRow)
{
  auto table = p::make_data_table("run", p::data_column<int>("x"));
  std::vector<std::string> events;
  table.attach(recording_sink{"a", &events, {}, false, false, [&] { table.append(2); }});
  EXPECT_THROW(table.append(1), p::data_delivery_error);
  EXPECT_EQ(table.size(), 1U);
  EXPECT_EQ(std::get<0>(table.rows()[0]), 1);
}

struct flush_failure_buffer : std::stringbuf
{
    int sync() override { return -1; }
};
TEST(DataTableStreaming, ExplicitFinishReportsDelayedStreamFailure)
{
  auto table = p::make_data_table("run", p::data_column<int>("x"));
  flush_failure_buffer buffer;
  std::ostream out(&buffer);
  table.attach(p::tsv_sink(out));
  table.append(1);
  EXPECT_THROW(table.finish(), p::data_delivery_error);
  EXPECT_EQ(buffer.str(), "x\n1\n");
}

TEST(DataTableStreaming, MoveTransfersSubscriptionsWithoutReplaying)
{
  auto source = p::make_data_table("run", p::data_column<int>("x"));
  std::ostringstream out;
  auto attachment = source.attach(p::csv_sink(out));
  source.append(1);
  auto destination = std::move(source);
  destination.append(2);
  destination.finish();
  EXPECT_EQ(destination.sink_state(attachment.id), p::data_sink_state::closed);
  EXPECT_EQ(out.str(), "x\n1\n2\n");
}
#if defined(__linux__)
TEST(DataTableStreaming, DefaultTerminalWriteFailureReachesRequiredSinkPolicy)
{
  EXPECT_EXIT(
      {
        if (!std::freopen("/dev/full", "w", stdout)) std::_Exit(2);
        uni20::display::reset_sink();
        auto table = p::make_data_table("run", p::data_column<int>("x"));
        table.attach(p::terminal_sink());
        try
        {
          table.append(1);
        }
        catch (p::data_delivery_error const&)
        {
          std::_Exit(0);
        }
        std::_Exit(1);
      },
      ::testing::ExitedWithCode(0), "");
}
#endif

} // namespace
