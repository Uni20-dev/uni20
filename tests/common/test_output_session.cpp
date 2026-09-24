#include <gtest/gtest.h>
#include <sstream>
#include <uni20/common/output_session.hpp>
#include <unistd.h>

using namespace uni20;
namespace p = uni20::presentation;
namespace
{
auto table() { return p::make_data_table("Results", p::data_column<int>("n"), p::data_column<long double>("energy")); }
metadata_document initial()
{
  metadata_document doc;
  doc.group("model");
  doc.add("model", "u", 4.L);
  return doc;
}
struct failing_buffer : std::streambuf
{
    bool fail_write = false;
    bool fail_flush = false;
    int flushes = 0;
    std::string contents;
    std::string const& str() const { return contents; }
    std::streamsize xsputn(char const* p, std::streamsize n) override
    {
      if (fail_write) return 0;
      contents.append(p, static_cast<std::size_t>(n));
      return n;
    }
    int_type overflow(int_type c) override
    {
      if (fail_write) return traits_type::eof();
      if (!traits_type::eq_int_type(c, traits_type::eof())) contents += traits_type::to_char_type(c);
      return traits_type::not_eof(c);
    }
    int sync() override
    {
      ++flushes;
      return fail_flush ? -1 : 0;
    }
};
struct test_stream : std::ostream
{
    failing_buffer buffer;
    test_stream() : std::ostream(&buffer) {}
};
struct directory
{
    std::filesystem::path path;
    directory()
    {
      std::string pattern = (std::filesystem::temp_directory_path() / "uni20-output-XXXXXX").string();
      auto* result = mkdtemp(pattern.data());
      if (!result) throw std::runtime_error("mkdtemp failed");
      path = result;
    }
    ~directory()
    {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
};
} // namespace

TEST(OutputSession, RetainedReplayProjectionAndStrictTsv)
{
  auto data = table();
  data.append(1, 1.000000000000000001L);
  auto out = std::make_shared<std::ostringstream>();
  output_session session(initial());
  session.stream("file", out, output_format::tsv, {.projection = {.columns = {"energy"}}});
  session.attach(data, "results");
  data.append(2, 2.L);
  data.finish({{"local", "done"}});
  session.finish();
  EXPECT_EQ(out->str(), "energy\n" + metadata_value(1.000000000000000001L).text({}, true) + "\n2\n");
  auto text = out->str();
  session.finish();
  EXPECT_EQ(out->str(), text);
}

TEST(OutputSession, NamedJsonSequentialDifferentSchemasAndFinalSummary)
{
  auto out = std::make_shared<std::ostringstream>();
  output_session session(initial());
  session.stream("json", out, output_format::named_json);
  auto first = table();
  first.append(1, -1.L);
  session.write_table(first, "spectrum", {{"local", "first"}});
  auto second = p::make_data_table("Status", p::data_column<bool>("converged"));
  second.append(true);
  session.write_table(second, "status");
  metadata_document summary;
  summary.group("summary");
  summary.add("summary", "outcome", "partial");
  session.finish(summary);
  auto text = out->str();
  EXPECT_TRUE(text.starts_with("{\"tables\":[{\"name\":\"spectrum\""));
  EXPECT_NE(text.find("},{\"name\":\"status\""), std::string::npos);
  EXPECT_TRUE(text.ends_with("],\"summary\":{\"outcome\":\"partial\"},\"status\":\"partial\"}"));
}

TEST(OutputSession, SnapshotExportLeavesSourceAndOtherSubscriptionsOpen)
{
  auto data = table();
  std::ostringstream live;
  data.attach(p::tsv_sink(live));
  data.append(1, 2.L);
  auto snapshot = std::make_shared<std::ostringstream>();
  output_session session;
  session.stream("snapshot", snapshot, output_format::json);
  session.write_table(data, "snapshot", {{"local", "snapshot only"}});
  EXPECT_FALSE(data.finished());
  auto saved = snapshot->str();
  EXPECT_NO_THROW(data.append(2, 3.L));
  EXPECT_EQ(snapshot->str(), saved);
  EXPECT_EQ(live.str(), "n\tenergy\n1\t2\n2\t3\n");
  data.finish({{"local", "original final summary"}});
  session.finish();
  EXPECT_EQ(snapshot->str(), saved);
  auto replay = std::make_shared<std::ostringstream>();
  output_session late;
  late.stream("late", replay, output_format::json);
  late.write_table(data, "finished");
  late.finish();
  EXPECT_NE(replay->str().find("original final summary"), std::string::npos);
  EXPECT_EQ(replay->str().find("snapshot only"), std::string::npos);
}

TEST(OutputSession, CommentMetadataEscapesControlsAndCanBeSuppressed)
{
  auto doc = initial();
  doc.add("model", "note", "line1\nline2\r\x1b");
  for (bool preamble : {false, true})
  {
    auto out = std::make_shared<std::ostringstream>();
    output_session session(doc);
    session.stream("commented", out, output_format::commented_tsv, {.preamble = preamble});
    auto data = table();
    data.append(1, 2.L);
    session.write_table(data, "results");
    session.finish(doc);
    if (preamble)
      EXPECT_NE(out->str().find("# note: line1\\x0aline2\\x0d\\x1b\n"), std::string::npos);
    else
      EXPECT_EQ(out->str(), "n\tenergy\n1\t2\n");
  }
}

TEST(OutputSession, RequiredRowFailureStillDeliversAndCleanupPreservesErrors)
{
  auto bad = std::make_shared<test_stream>();
  auto good = std::make_shared<std::ostringstream>();
  output_session session;
  int closes = 0;
  session.stream("bad", bad, output_format::tsv, {}, [&] {
    ++closes;
    throw std::runtime_error("close failed");
  });
  session.stream("good", good, output_format::tsv);
  auto data = table();
  session.attach(data, "results");
  bad->buffer.fail_write = true;
  EXPECT_THROW(data.append(1, 2.L), p::data_delivery_error);
  EXPECT_EQ(data.size(), 1);
  EXPECT_EQ(good->str(), "n\tenergy\n1\t2\n");
  EXPECT_NO_THROW(data.append(2, 3.L)); // Failed sink is disabled; no retry of row 1.
  data.finish();
  EXPECT_THROW(session.finish(), output_error);
  EXPECT_EQ(closes, 1);
  EXPECT_GE(session.report().failures.size(), 2);
  EXPECT_EQ(session.report().failures.front().operation, output_operation::row);
  EXPECT_EQ(session.report().failures.back().operation, output_operation::close);
  EXPECT_THROW(session.finish(), output_error);
  EXPECT_EQ(closes, 1);
}

TEST(OutputSession, OptionalFailureReportedAndRequiredBeginAttemptsAll)
{
  for (bool required : {false, true})
  {
    auto bad = std::make_shared<test_stream>();
    bad->buffer.fail_write = true;
    auto good = std::make_shared<std::ostringstream>();
    output_session session;
    session.stream("bad", bad, output_format::tsv, {.required = required});
    session.stream("good", good, output_format::tsv);
    auto data = table();
    data.append(1, 2.L);
    if (required)
      EXPECT_THROW(session.attach(data, "results"), output_error);
    else
      EXPECT_EQ(session.attach(data, "results").failures.size(), 1);
    EXPECT_EQ(good->str(), "n\tenergy\n1\t2\n");
    data.finish();
    if (required)
      EXPECT_THROW(session.finish(), output_error);
    else
      EXPECT_FALSE(session.finish().failures.empty());
  }
}

TEST(OutputSession, FlushDoesNotFinishAndFailureDisablesFurtherWrites)
{
  auto bad = std::make_shared<test_stream>();
  auto good = std::make_shared<test_stream>();
  output_session session;
  session.stream("bad", bad, output_format::tsv);
  session.stream("good", good, output_format::tsv);
  auto data = table();
  session.attach(data, "results");
  data.append(1, 2.L);
  session.flush();
  EXPECT_FALSE(data.finished());
  EXPECT_EQ(good->buffer.flushes, 1);
  bad->buffer.fail_flush = true;
  EXPECT_THROW(session.flush(), output_error);
  EXPECT_EQ(good->buffer.flushes, 2);
  EXPECT_THROW(data.append(2, 3.L), p::data_delivery_error);
  EXPECT_EQ(bad->buffer.str(), "n\tenergy\n1\t2\n");
  EXPECT_EQ(good->buffer.str(), "n\tenergy\n1\t2\n2\t3\n");
  data.finish();
  EXPECT_THROW(session.finish(), output_error);
}

TEST(OutputSession, ResourcesSurviveSessionAndCallerOwners)
{
  auto data = table();
  std::weak_ptr<std::ostream> weak;
  {
    auto out = std::make_shared<std::ostringstream>();
    weak = out;
    output_session session;
    session.stream("owned", out, output_format::tsv);
    session.attach(data, "results");
  }
  ASSERT_FALSE(weak.expired());
  EXPECT_NO_THROW(data.append(1, 2.L));
  EXPECT_NO_THROW(data.finish());
  EXPECT_TRUE(weak.expired());
}

TEST(OutputSession, LateSinksReplayOriginalMetadataAndNoHistoryNeedsExplicitPolicy)
{
  auto doc = initial();
  auto data = table();
  data.append(1, 2.L);
  output_session session(doc);
  doc.replace("u", 8.L);
  auto out = std::make_shared<std::ostringstream>();
  session.stream("late", out, output_format::json);
  session.attach(data, "results");
  data.finish();
  session.finish();
  EXPECT_NE(out->str().find("\"u\":\"4\""), std::string::npos);
  auto no_history = p::make_data_table("stream", {.retain = p::retention::none}, p::data_column<int>("n"));
  no_history.append(1);
  auto future = std::make_shared<std::ostringstream>();
  output_session other;
  other.stream("future", future, output_format::json);
  EXPECT_THROW(other.attach(no_history, "results"), std::logic_error);
  EXPECT_TRUE(future->str().empty());
  other.attach(no_history, "results", p::sink_replay::future_only);
  no_history.append(2);
  no_history.finish();
  other.finish();
  EXPECT_NE(future->str().find("\"first_row\":1"), std::string::npos);
}

TEST(OutputSession, PlainOutputIsLocalAndPreservesLongNumericTokens)
{
  int global_events = 0;
  display::scoped_sink global([&](auto const& event) {
    ++global_events;
    EXPECT_EQ(event.destination, display::stream::err);
  });
  auto out = std::make_shared<std::ostringstream>();
  output_session session(initial());
  auto policy = p::plain_policy();
  policy.wrap_width = 20;
  session.stream("terminal", out, output_format::terminal, {.human_policy = policy});
  auto data = table();
  data.append(1, 1.000000000000000001L);
  session.write_table(data, "results");
  session.finish();
  display::emit(p::styled_text{}.append("error"), display::stream::err);
  EXPECT_EQ(global_events, 1);
  EXPECT_NE(out->str().find(metadata_value(1.000000000000000001L).text()), std::string::npos);
}

TEST(OutputSession, HumanPreambleAndRowsAreFlushedBeforeCalculationContinues)
{
  auto out = std::make_shared<test_stream>();
  output_session session(initial());
  session.stream("live", out, output_format::terminal);
  session.open();
  EXPECT_EQ(out->buffer.flushes, 1);
  EXPECT_FALSE(out->buffer.str().empty());
  auto data = table();
  session.attach(data, "results");
  data.append(1, 2.L);
  EXPECT_EQ(out->buffer.flushes, 2);
  data.finish();
  session.finish();
}

TEST(OutputSession, ExclusiveFilesAliasesAndOverwrite)
{
  directory dir;
  auto path = dir.path / "results.tsv";
  {
    std::ofstream file(path);
    file << "original";
  }
  output_session no_overwrite;
  no_overwrite.file(path, output_format::tsv);
  EXPECT_THROW(no_overwrite.open(), output_error);
  std::ifstream existing(path);
  std::string text;
  existing >> text;
  EXPECT_EQ(text, "original");
  auto alias = dir.path / "alias.tsv";
  std::filesystem::create_hard_link(path, alias);
  output_session conflicting;
  conflicting.file(path, output_format::tsv, {.overwrite = true});
  conflicting.file(alias, output_format::tsv, {.overwrite = true});
  EXPECT_THROW(conflicting.open(), std::invalid_argument);
  auto link = dir.path / "link.tsv";
  std::filesystem::create_symlink(path, link);
  output_session symlink;
  symlink.file(path, output_format::tsv, {.overwrite = true});
  symlink.file(link, output_format::tsv, {.overwrite = true});
  EXPECT_THROW(symlink.open(), std::invalid_argument);
  output_session overwrite;
  overwrite.file(path, output_format::tsv, {.overwrite = true});
  auto data = table();
  data.append(1, 2.L);
  overwrite.write_table(data, "results");
  overwrite.finish();
  std::ifstream changed(path);
  std::string header;
  std::getline(changed, header);
  EXPECT_EQ(header, "n\tenergy");
}

TEST(OutputSession, QuietSuppressesMachineStdoutButRetainsFiles)
{
  directory dir;
  output_session session({}, true);
  session.standard_output(output_format::json);
  auto path = dir.path / "results.tsv";
  session.file(path, output_format::tsv);
  auto data = table();
  data.append(1, 2.L);
  session.write_table(data, "results");
  session.finish();
  EXPECT_GT(std::filesystem::file_size(path), 0);
}

TEST(OutputSession, StructuralMisuseAndMetadataCollisions)
{
  auto out = std::make_shared<std::ostringstream>();
  output_session session(initial());
  session.stream("first", out, output_format::tsv);
  session.stream("alias", out, output_format::json);
  EXPECT_THROW(session.open(), std::invalid_argument);
  EXPECT_TRUE(out->str().empty());
  output_session active;
  active.stream("active", out, output_format::json);
  auto data = table();
  active.attach(data, "results");
  EXPECT_THROW(active.finish(), output_error);
  output_session duplicate(initial());
  auto other = std::make_shared<std::ostringstream>();
  duplicate.stream("json", other, output_format::json);
  auto collision = p::make_data_table("collision", {.metadata = {{"u", "8"}}}, p::data_column<int>("n"));
  EXPECT_THROW(duplicate.attach(collision, "results"), output_error);
  EXPECT_TRUE(other->str().empty());
}
