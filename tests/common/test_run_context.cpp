#include <gtest/gtest.h>
#include <uni20/common/configuration.hpp>
#include <uni20/common/run_context.hpp>
#include <uni20/common/terminal.hpp>

using namespace uni20;

TEST(Metadata, OwnsTextAndRetainsNativeTypes)
{
  std::string text = "original";
  metadata_value value{std::string_view(text)};
  text = "changed";
  EXPECT_EQ(value.get<std::string>(), "original");
  auto high = parse_real<long double>("1.000000000000000001");
  metadata_value native(high);
  EXPECT_EQ(native.type(), typeid(long double));
  EXPECT_EQ(parse_real<long double>(native.text({}, true)), high);
  EXPECT_THROW((void)native.get<double>(), std::bad_any_cast);
  metadata_value missing(std::optional<int>{});
  EXPECT_TRUE(missing.missing());
  EXPECT_EQ(missing.type(), typeid(std::optional<int>));
  EXPECT_FALSE(missing.as<std::optional<long>>());
  EXPECT_THROW((void)missing.as<int>(), std::invalid_argument);
}

TEST(Metadata, CheckedConversions)
{
  EXPECT_THROW((void)metadata_value(-1).as<unsigned>(), std::overflow_error);
  EXPECT_THROW((void)metadata_value(256).as<unsigned char>(), std::overflow_error);
  EXPECT_THROW((void)metadata_value("18446744073709551616").as<unsigned long long>(), std::out_of_range);
  EXPECT_THROW((void)metadata_value("12x").as<int>(), std::invalid_argument);
  EXPECT_THROW((void)metadata_value(1.5).as<int>(), std::invalid_argument);
  EXPECT_THROW((void)metadata_value(1.5).as<half_int>(), std::invalid_argument);
  EXPECT_EQ(metadata_value("3/2").as<half_int>(), half_int::parse("1.5"));
  EXPECT_EQ(metadata_value(3).as<half_int>(), half_int(3));
  EXPECT_EQ(metadata_value(basic_half_int<short>::parse("7/2")).as<half_int>(), half_int::parse("3.5"));
  EXPECT_THROW((void)metadata_value(half_int(64)).as<basic_half_int<signed char>>(), std::overflow_error);
  EXPECT_THROW((void)metadata_value(numeric_limits<double>::max()).as<float>(), std::out_of_range);
  EXPECT_FALSE(metadata_value("false").as<bool>());
  EXPECT_EQ(metadata_value(std::optional<int>(0)).as<long>(), 0);
  EXPECT_THROW((void)metadata_value(static_cast<char const*>(nullptr)), std::invalid_argument);
}

TEST(Metadata, OrderedSnapshotsReplacementAndExportMapping)
{
  metadata_document doc;
  doc.group("parameters", "Parameters");
  doc.add("parameters", "spin", half_int::parse("3/2"), {.label = "Spin", .display = {.fractions = true}});
  doc.add("parameters", "n", 7);
  auto snapshot = doc;
  doc.replace("n", 8);
  EXPECT_EQ(snapshot.find("n")->value.get<int>(), 7);
  EXPECT_EQ(doc.find("n")->value.get<int>(), 8);
  EXPECT_EQ(doc.groups().front().fields.front().id, "spin");
  EXPECT_THROW((void)doc.replace("n", 8L), std::invalid_argument);
  EXPECT_THROW((void)doc.add("parameters", "n", 8), std::invalid_argument);
  EXPECT_THROW((void)doc.group("parameters"), std::invalid_argument);
  EXPECT_EQ(doc.strings({{"spin", "S"}}).at("S"), "1.5");
  EXPECT_THROW((void)doc.strings({{"spin", "n"}}), std::invalid_argument);
  EXPECT_THROW((void)doc.strings({{"unknown", "x"}}), std::invalid_argument);
  auto rendered = presentation::render_plain(metadata_report(doc));
  EXPECT_NE(rendered.find("3/2"), std::string::npos);
}

TEST(Configuration, EveryPrecedenceStepAndOrigin)
{
  auto reader = [](std::string const&) -> std::optional<std::string> { return "2"; };
  configuration c(reader);
  c.add("count", 1).environment("COUNT").attribute("input.wf", "Count").file("count");
  EXPECT_EQ(c.get<int>("count"), 2);
  EXPECT_EQ(c.resolve("count")->origin.source, configuration_source::environment);
  c.attributes("input.wf", {{"Count", 3}});
  EXPECT_EQ(c.get<int>("count"), 3);
  EXPECT_EQ(c.resolve("count")->origin.location, "input.wf");
  c.option_file("job.toml", {{"count", "4"}});
  EXPECT_EQ(c.get<int>("count"), 4);
  EXPECT_EQ(c.resolve("count")->origin.location, "job.toml");
  c.set("count", 5);
  EXPECT_EQ(c.get<int>("count"), 5);
  EXPECT_EQ(c.resolve("count")->origin.source, configuration_source::explicit_input);
  c.add("default", 17);
  EXPECT_EQ(c.get<int>("default"), 17);
  EXPECT_EQ(c.resolve("default")->origin.source, configuration_source::application_default);
}

TEST(Configuration, PresenceIsNotTruthinessAndInvalidWinnersFail)
{
  configuration c([](auto const&) -> std::optional<std::string> { return ""; });
  c.add("name", std::string("default")).environment("NAME");
  c.add("flag", true);
  c.add("count", 12).file("count");
  EXPECT_EQ(c.get<std::string>("name"), "");
  c.set("flag", false);
  c.set("count", 0);
  EXPECT_FALSE(c.get<bool>("flag"));
  EXPECT_EQ(c.get<int>("count"), 0);
  c.set("count", "invalid");
  EXPECT_THROW((void)c.get<int>("count"), std::invalid_argument);
  EXPECT_THROW((void)c.option_file("bad", {{"undeclared", 5}}), std::invalid_argument);
}

TEST(Configuration, RequiredAttributesBootstrapAndExplicitCopy)
{
  configuration c;
  c.add("input", std::string("state.wf"));
  c.add<std::string>("hamiltonian").required().attribute("wavefunction", "Hamiltonian");
  EXPECT_EQ(c.get<std::string>("input"), "state.wf");
  EXPECT_THROW((void)c.snapshot(), std::invalid_argument);
  attribute_map input{{"Hamiltonian", "H"}, {"private_note", "not selected"}};
  c.attributes("wavefunction", input);
  auto doc = c.snapshot();
  EXPECT_EQ(doc.find("hamiltonian")->value.get<std::string>(), "H");
  EXPECT_EQ(doc.find("private_note"), nullptr);
  attribute_map output;
  c.copy_attributes(output, {{"hamiltonian", "Hamiltonian"}});
  EXPECT_EQ(output.size(), 1);
  EXPECT_EQ(input.size(), 2);
  c.add("steps", 4).check<int>([](int n) {
    if (n < 1) throw std::invalid_argument("positive steps required");
  });
  c.set("steps", 0);
  EXPECT_THROW((void)c.snapshot(), std::invalid_argument);
}

TEST(Configuration, DeferredPrecisionAndNativeValues)
{
  configuration c;
  c.add("u", std::string("1.000000000000000001"));
  auto value = c.resolve("u")->value.as<long double>();
  EXPECT_EQ(value, parse_real<long double>("1.000000000000000001"));
  c.add("native", value);
  EXPECT_EQ(c.get<long double>("native"), value);
#if UNI20_HAS_FLOAT128
  auto quad = parse_real<float128>("1.000000000000000000000000000001");
  c.add("quad", quad);
  EXPECT_EQ(c.get<float128>("quad"), quad);
  EXPECT_EQ(parse_real<float128>(c.snapshot().strings().at("quad")), quad);
#endif
}

TEST(RunContext, DeterministicDisjointTimingAndFrozenSnapshots)
{
  run_clock_sample now{10, 5};
  run_context run({.name = "solver"}, {.clock = [&] { return now; }, .utc = [] { return "fixed UTC"; }});
  run.metadata().group("model");
  run.metadata().add("model", "u", 4.L);
  auto before = run.snapshot();
  run.metadata().replace("u", 5.L);
  now = {12, 6};
  EXPECT_EQ(run.measure([&] {
    now = {15, 9};
    return 42;
  }),
            42);
  now = {17, 11}; // Includes rendering CPU outside the marked computation.
  {
    auto scope = run.computation();
    now = {19, 12};
  }
  now = {20, 14};
  auto const& summary = run.finish(run_outcome::partial);
  EXPECT_EQ(before.find("u")->value.get<long double>(), 4.L);
  EXPECT_EQ(summary.find("compute_cpu_seconds")->value.as<long double>(), 4);
  EXPECT_EQ(summary.find("run_cpu_seconds")->value.as<long double>(), 9);
  EXPECT_EQ(summary.find("elapsed_seconds")->value.as<long double>(), 10);
  EXPECT_EQ(summary.find("outcome")->value.get<std::string>(), "partial");
  EXPECT_THROW((void)run.metadata(), std::logic_error);
  EXPECT_THROW((void)run.computation(), std::logic_error);
  EXPECT_THROW((void)run.finish(run_outcome::success), std::logic_error);
}

TEST(RunContext, ScopeMisuseExceptionsAndUnavailableClock)
{
  run_clock_sample now{0, 0};
  run_context run({.name = "solver"}, {.clock = [&] { return now; }});
  {
    auto scope = run.computation();
    EXPECT_THROW((void)run.computation(), std::logic_error);
    EXPECT_THROW((void)run.finish(run_outcome::success), std::logic_error);
    now = {1, std::nullopt};
  }
  EXPECT_THROW((void)run.measure([] { throw std::runtime_error("solver failure"); }), std::runtime_error);
  auto const& summary = run.finish(run_outcome::failed);
  EXPECT_TRUE(summary.find("compute_cpu_seconds")->value.missing());
  EXPECT_TRUE(summary.find("run_cpu_seconds")->value.missing());
}

TEST(RunContext, InvocationIsOptInOwnedAndEscapedAtPresentation)
{
  run_context omitted({.name = "solver"});
  EXPECT_EQ(omitted.snapshot().find("invocation"), nullptr);
  EXPECT_EQ(presentation::render_plain(metadata_report(omitted.snapshot())).find("Build"), std::string::npos);
  EXPECT_NE(presentation::render_plain(metadata_report(omitted.snapshot(), {}, true)).find("Build"), std::string::npos);
  run_context run({.name = "solver"}, {.invocation = {"solver", "", "$HOME", "a\nb", "a'b"}});
  auto snapshot = run.snapshot();
  EXPECT_EQ(run.invocation().size(), 5);
  EXPECT_NE(snapshot.find("invocation")->value.text().find("\\$HOME"), std::string::npos);
  EXPECT_EQ(metadata_line("one\ntwo\x1b"), "one\\x0atwo\\x1b");
  EXPECT_EQ(terminal::quote_shell(""), "\"\"");
  EXPECT_EQ(terminal::quote_shell("`id`"), "\"\\`id\\`\"");
  auto rendered = presentation::render_plain(metadata_report(snapshot, {}, true));
  EXPECT_NE(rendered.find("a\\x0ab"), std::string::npos);
}
