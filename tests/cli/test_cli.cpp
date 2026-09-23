#include <gtest/gtest.h>
#include <uni20/cli/cli.hpp>
#include <uni20/core/scalar_io.hpp>

#include <limits>

namespace
{
namespace c = uni20::cli;
namespace p = uni20::presentation;

c::parse_result parse(CLI::App& app, std::initializer_list<char const*> args, c::parse_policy policy = {})
{
  std::vector<char const*> argv{"probe"};
  argv.insert(argv.end(), args);
  return c::parse(app, static_cast<int>(argv.size()), argv.data(), policy);
}

p::program_info program()
{
  return {.name = "probe",
          .description = "Inspect model parameters",
          .version = "1.2",
          .revision = "abc123",
          .examples = {{"probe --spin=-3/2", "Exact quantum number"}},
          .notes = {"No numerical solver is started by informational requests."}};
}

std::string render(p::report_builder const& report) { return p::render_plain(report, p::strict_ascii_policy()); }
} // namespace

TEST(CliActions, EmptyInvocationPolicy)
{
  CLI::App app;
  c::configure(app, program());
  auto result = parse(app, {});
  EXPECT_EQ(result.requested, c::action::help);
  EXPECT_EQ(result.destination, uni20::display::stream::err);
  EXPECT_EQ(result.exit_code, 1);
  result = parse(app, {}, {.empty_destination = uni20::display::stream::out, .empty_exit_code = 0});
  EXPECT_EQ(result.destination, uni20::display::stream::out);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(parse(app, {}, {.empty = c::no_arguments::run}).requested, c::action::run);
}

TEST(CliActions, InformationPrecedesRequirementsValidationAndOrdinaryCallbacks)
{
  for (auto const& [flag, expected] : {std::pair{"--help", c::action::help},
                                       {"-h", c::action::help},
                                       {"--version", c::action::version},
                                       {"--build-info", c::action::build_info}})
  {
    SCOPED_TRACE(flag);
    CLI::App app;
    c::configure(app, program());
    std::string required, precision;
    bool called = false;
    app.add_option("--input", required)->required();
    app.add_option("--precision", precision)->check(CLI::IsMember({"fp64", "long-double"}));
    app.add_flag_callback("--calculate", [&] { called = true; });
    auto const result = parse(app, {"--calculate", "--precision=unavailable", flag});
    EXPECT_EQ(result.requested, expected);
    EXPECT_EQ(result.destination, uni20::display::stream::out);
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_FALSE(called);
  }
}

TEST(CliActions, LiteralHelpTokensRemainValues)
{
  CLI::App app;
  c::configure(app, program());
  std::string positional, named;
  app.add_option("input", positional);
  app.add_option("--label", named);
  EXPECT_EQ(parse(app, {"--label=--version", "--", "--help"}).requested, c::action::run);
  EXPECT_EQ(positional, "--help");
  EXPECT_EQ(named, "--version");
}

TEST(CliActions, ErrorsAreOwnedConciseAndDoNotRequestReferences)
{
  CLI::App app;
  auto info = program();
  unsigned references = 0;
  info.references = [&] {
    ++references;
    return std::vector<p::program_reference>{{.key = "citation", .citation = "A full literature reference"}};
  };
  c::configure(app, info);
  auto const result = parse(app, {"--unknown"}, {.error_exit_code = 2});
  EXPECT_EQ(result.requested, c::action::error);
  EXPECT_EQ(result.exit_code, 2);
  EXPECT_EQ(result.destination, uni20::display::stream::err);
  auto text = render(c::result_report(app, info, result));
  EXPECT_NE(text.find("--unknown"), std::string::npos);
  EXPECT_NE(text.find("probe --help"), std::string::npos);
  EXPECT_EQ(text.find("References"), std::string::npos);
  EXPECT_EQ(references, 0u);
  EXPECT_EQ(render(c::result_report(app, info, {.requested = c::action::version})), "probe 1.2 (abc123)\n");
  EXPECT_EQ(references, 0u);
  EXPECT_NE(render(c::result_report(app, info, {.requested = c::action::help})).find("citation"), std::string::npos);
  EXPECT_EQ(references, 1u);
  EXPECT_THROW(static_cast<void>(c::result_report(app, info, {})), std::logic_error);
}

TEST(CliActions, BuildInformationUsesConfiguredUni20Snapshot)
{
  auto const text = render(c::build_info_report(program()));
  EXPECT_NE(text.find("probe 1.2 (abc123)"), std::string::npos);
  EXPECT_NE(text.find("Compiler"), std::string::npos);
  EXPECT_NE(text.find("UNI20_BUILD_CLI"), std::string::npos);
  EXPECT_NE(text.find("CLI11"), std::string::npos);
}

TEST(CliConversions, CountsRejectNegativeOverflowAndPartialTokens)
{
  for (auto const* token : {"-1", "18446744073709551616", "7x", "1.0", ""})
  {
    SCOPED_TRACE(token);
    CLI::App app;
    std::uint64_t value = 19;
    c::add_count_option(app, "--count", value);
    EXPECT_EQ(parse(app, {"--count", token}).requested, c::action::error);
    EXPECT_EQ(value, 19u);
  }
  CLI::App app;
  std::uint64_t value = 0;
  c::add_count_option(app, "--count", value);
  EXPECT_EQ(parse(app, {"--count=18446744073709551615"}).requested, c::action::run);
  EXPECT_EQ(value, std::numeric_limits<std::uint64_t>::max());
}

TEST(CliConversions, HalfIntegersAreExactAndRespectBoundStorageRange)
{
  for (auto const* token : {"-3/2", "-1.5"})
  {
    CLI::App app;
    uni20::half_int spin{};
    c::add_half_int_option(app, "--spin", spin);
    EXPECT_EQ(parse(app, {"--spin", token}).requested, c::action::run);
    EXPECT_EQ(spin, uni20::from_twice(-3));
  }
  for (auto const* token : {"0.25", "1/3", "128", "-64.5", "99999999999999999999999999"})
  {
    SCOPED_TRACE(token);
    CLI::App app;
    uni20::basic_half_int<signed char> spin(1);
    c::add_half_int_option(app, "--spin", spin);
    EXPECT_EQ(parse(app, {"--spin", token}).requested, c::action::error);
    EXPECT_EQ(spin, uni20::basic_half_int<signed char>(1));
  }
}

TEST(CliConversions, HalfIntegerLowerBoundaryReportsValidationErrorWithoutAssignment)
{
  CLI::App app;
  uni20::half_int spin(1);
  c::add_half_int_option(app, "--spin", spin);
  auto const result = parse(app, {"--spin=-4611686018427387904.5"});
  EXPECT_EQ(result.requested, c::action::error);
  EXPECT_EQ(result.destination, uni20::display::stream::err);
  EXPECT_NE(result.message.find("--spin"), std::string::npos);
  EXPECT_NE(result.message.find("overflow"), std::string::npos);
  EXPECT_EQ(spin, uni20::half_int(1));
}

TEST(CliDefaults, ExplicitDefaultsAssignBoundValuesAndAllowCommandLineOverrides)
{
  CLI::App app;
  unsigned count = 1;
  uni20::half_int spin(1);
  auto* count_option = c::add_count_option(app, "--count", count)->default_val(12);
  auto* spin_option = c::add_half_int_option(app, "--spin", spin)->default_val("1.5");
  EXPECT_EQ(count, 12u); // Native variable bindings apply default_val during registration.
  EXPECT_EQ(spin, uni20::from_twice(3));
  EXPECT_EQ(parse(app, {}, {.empty = c::no_arguments::run}).requested, c::action::run);
  EXPECT_EQ(count, 12u);
  EXPECT_EQ(spin, uni20::from_twice(3));
  EXPECT_EQ(count_option->get_default_str(), "12");
  EXPECT_EQ(spin_option->get_default_str(), "1.5");
  EXPECT_EQ(parse(app, {"--count=7", "--spin=-3/2"}).requested, c::action::run);
  EXPECT_EQ(count, 7u);
  EXPECT_EQ(spin, uni20::from_twice(-3));
}

TEST(CliDefaults, CountDefaultsUseCheckedConversionAndRegisteredValidators)
{
  CLI::App app;
  unsigned count = 1;
  auto* option = c::add_count_option(app, "--count", count)->capture_default_str();
  for (auto const* invalid : {"-1", "18446744073709551616", "12x"})
  {
    SCOPED_TRACE(invalid);
    EXPECT_THROW(option->default_val(invalid), CLI::ValidationError);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(option->get_default_str(), "1");
  }
  option->check(CLI::PositiveNumber);
  EXPECT_THROW(option->default_val(0), CLI::ValidationError);
  EXPECT_EQ(count, 1u);
}

TEST(CliDefaults, HalfIntegerDefaultsUseExactConversionAndRegisteredValidators)
{
  CLI::App app;
  uni20::half_int spin(1);
  auto* option = c::add_half_int_option(app, "--spin", spin)->capture_default_str();
  for (auto const* invalid : {"0.25", "1/3", "-4611686018427387904.5", "99999999999999999999999999"})
  {
    SCOPED_TRACE(invalid);
    EXPECT_THROW(option->default_val(invalid), CLI::ValidationError);
    EXPECT_EQ(spin, uni20::half_int(1));
    EXPECT_EQ(option->get_default_str(), "1");
  }
  option->check(CLI::IsMember({"1", "1.5"}));
  EXPECT_THROW(option->default_val("2.5"), CLI::ValidationError);
  EXPECT_EQ(spin, uni20::half_int(1));
}

TEST(CliDefaults, InitializedVariablesAreTheSingleSourceForCapturedDefaults)
{
  CLI::App app;
  c::configure(app, program());
  unsigned count = 12;
  uni20::half_int spin = uni20::from_twice(3);
  auto* count_option = c::add_count_option(app, "--count", count)->capture_default_str();
  auto* spin_option = c::add_half_int_option(app, "--spin", spin)->capture_default_str();
  EXPECT_EQ(parse(app, {}, {.empty = c::no_arguments::run}).requested, c::action::run);
  EXPECT_EQ(count, 12u);
  EXPECT_EQ(spin, uni20::from_twice(3));
  EXPECT_EQ(count_option->get_default_str(), "12");
  EXPECT_EQ(spin_option->get_default_str(), "1.5");
  auto const text = render(c::help_report(app, program()));
  EXPECT_NE(text.find("default: 12"), std::string::npos);
  EXPECT_NE(text.find("default: 1.5"), std::string::npos);
  EXPECT_EQ(parse(app, {"--count=5", "--spin=7/2"}).requested, c::action::run);
  EXPECT_EQ(count, 5u);
  EXPECT_EQ(spin, uni20::from_twice(7));
  // The displayed defaults describe registration, not the most recent arguments.
  EXPECT_EQ(count_option->get_default_str(), "12");
  EXPECT_EQ(spin_option->get_default_str(), "1.5");
}

TEST(CliOptions, NativeDeclarationsEnforceRepetitionAndRelationships)
{
  CLI::App app;
  c::configure(app, program());
  std::vector<std::string> exports;
  std::string momentum;
  unsigned points = 1;
  auto* p = c::add_count_option(app, "--points", points);
  auto* m = app.add_option("--momentum", momentum)->excludes(p);
  app.add_option("--export", exports)->expected(1)->take_all()->needs(m);
  EXPECT_EQ(parse(app, {"--momentum=-0.5", "--export=a.tsv", "--export", "b.json"}).requested, c::action::run);
  EXPECT_EQ(exports, (std::vector<std::string>{"a.tsv", "b.json"}));
  EXPECT_EQ(momentum, "-0.5");
  EXPECT_EQ(parse(app, {"--points=3", "--momentum=1"}).requested, c::action::error);
  EXPECT_EQ(parse(app, {"--export=a.tsv"}).requested, c::action::error);
  EXPECT_EQ(parse(app, {"--points=2", "--points=3"}).requested, c::action::error);
}

TEST(CliHelp, DescribesActualOptionsGroupsDefaultsAndConstraints)
{
  CLI::App app;
  c::configure(app, program());
  unsigned points = 12;
  uni20::half_int spin = uni20::from_twice(3);
  std::string input, hidden, momentum, precision;
  auto* group = app.add_option_group("Model", "Model quantum numbers and input");
  group->require_option(1);
  c::add_half_int_option(*group, "-s,--spin", spin, "Exact spin")->capture_default_str();
  group->add_option("input", input)->required();
  auto* count =
      c::add_count_option(app, "--points", points, "Number of points")->capture_default_str()->group("Numerics");
  app.add_option("--momentum", momentum)->excludes(count)->group("Numerics");
  app.add_option("--precision", precision)->check(CLI::IsMember({"fp64", "fp128"}))->group("Numerics");
  app.add_option("--hidden", hidden)->group("");
  app.add_option_group("")->add_option("--hidden-group", hidden);
  app.add_option_group("+internal")->add_option("--internal", hidden);
  auto text = render(c::help_report(app, program()));
  for (auto const* part :
       {"Model", "Numerics", "-s,--spin", "HALF_INT", "default: 1.5", "default: 12", "required", "excludes: --points",
        "fp64", "fp128", "probe [options] input", "Model quantum numbers and input", "Minimum selected options: 1"})
    EXPECT_NE(text.find(part), std::string::npos) << part << '\n' << text;
  EXPECT_EQ(text.find("--hidden"), std::string::npos);
  EXPECT_EQ(text.find("--internal"), std::string::npos);
  count->description("Updated description");
  EXPECT_NE(render(c::help_report(app, program())).find("Updated description"), std::string::npos);
  auto policy = p::strict_ascii_policy();
  policy.color = p::color_mode::always;
  EXPECT_NE(p::render_terminal(c::help_report(app, program()), policy).find("\033["), std::string::npos);
  EXPECT_EQ(text.find("\033["), std::string::npos);
}

TEST(CliPrecision, TokensSurviveParsingUntilPrecisionIsResolved)
{
  for (bool precision_first : {false, true})
  {
    CLI::App app;
    std::string token, precision = "fp64";
    app.add_option("--value", token);
    app.add_option("--precision", precision);
    char const* exact = "1.000000000000000000000000000001";
    auto result = precision_first ? parse(app, {"--precision=fp128", "--value", exact})
                                  : parse(app, {"--value", exact, "--precision=fp128"});
    ASSERT_EQ(result.requested, c::action::run);
    EXPECT_EQ(token, exact);
    EXPECT_EQ(precision, "fp128");
#if UNI20_HAS_FLOAT128
    auto const value = uni20::parse_real<uni20::float128>(token);
    EXPECT_GT(value, uni20::float128(1));
    EXPECT_EQ(uni20::parse_real<uni20::float128>(uni20::format_real(value)), value);
#endif
  }
}
