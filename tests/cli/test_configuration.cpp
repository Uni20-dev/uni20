#include <fstream>
#include <gtest/gtest.h>
#include <uni20/cli/configuration.hpp>
#include <uni20/cli/output.hpp>
#include <unistd.h>

namespace c = uni20::cli;
namespace
{
c::parse_result parse(c::configuration& config, std::initializer_list<char const*> args)
{
  std::vector<char const*> argv{"probe"};
  argv.insert(argv.end(), args);
  return config.parse(static_cast<int>(argv.size()), argv.data(), {.empty = c::no_arguments::run});
}

TEST(CliConfiguration, TransformsRunOnceOnExplicitInputsAndSupportTextFallbacks)
{
  for (bool explicit_cli : {true, false})
  {
    CLI::App app;
    c::configure(app, {.name = "probe"});
    std::string text = "a";
    int count = 1;
    auto* option = app.add_option("--text", text)->transform([](std::string input) { return input + "!"; });
    auto* number =
        app.add_option("--count", count)->transform(CLI::CheckedTransformer(std::map<std::string, int>{{"two", 2}}));
    c::configuration config(app);
    config.bind(option, text, "text").attribute("state", "Text");
    config.bind(number, count, "count").attribute("state", "Count");
    auto result = explicit_cli ? parse(config, {"--text=x", "--count=two"}) : parse(config, {});
    ASSERT_EQ(result.requested, c::action::run);
    config.values().attributes("state", {{"Text", "x"}, {"Count", "two"}});
    auto first = config.resolve();
    EXPECT_EQ(text, "x!");
    EXPECT_EQ(count, 2);
    text = "caller changed this";
    count = 99;
    auto second = config.resolve();
    EXPECT_EQ(first.find("text")->value.text(), second.find("text")->value.text());
    EXPECT_EQ(text, "x!");
    EXPECT_EQ(count, 2);
  }
}

TEST(CliConfiguration, InitializedDefaultsAreValidatedAndNativeAttributesStayPrecise)
{
  CLI::App app;
  c::configure(app, {.name = "probe"});
  unsigned count = 0;
  std::string real = "1";
  bool enabled = true;
  c::configuration config(app);
  config.bind(c::add_count_option(app, "--count", count)->check(CLI::PositiveNumber), count, "count");
  config.bind(app.add_option("--real", real), real, "real").attribute("state", "Real");
  config.bind(app.add_flag("--enabled", enabled), enabled, "enabled").attribute("state", "Enabled");
  ASSERT_EQ(parse(config, {}).requested, c::action::run);
  EXPECT_THROW((void)config.resolve(), std::invalid_argument);
  auto native = uni20::parse_real<long double>("1.000000000000000001");
  config.values().set("count", 2);
  config.values().attributes("state", {{"Real", native}, {"Enabled", false}});
  auto result = config.resolve();
  EXPECT_FALSE(enabled);
  EXPECT_EQ(uni20::parse_real<long double>(real), native);
}

TEST(CliOutput, CommonControlsAreIndependentOfRunMetadataAndOpenNoFilesDuringHelp)
{
  CLI::App app;
  c::configure(app, {.name = "probe"});
  c::output_options options;
  c::add_output_options(app, options);
  char const* argv[]{"probe", "--help", "--output=/nonexistent/results.tsv", "--format=tsv"};
  EXPECT_EQ(c::parse(app, 4, argv).requested, c::action::help);
  options.format = "named-json";
  options.quiet = true;
  auto session = options.session();
  EXPECT_NO_THROW(session.finish());
  options.format = "invalid";
  EXPECT_THROW((void)options.session(), std::invalid_argument);
}
} // namespace

TEST(CliConfiguration, DefaultsAttributesEnvironmentAndCliUseOriginalValidators)
{
  for (bool explicit_cli : {false, true})
  {
    CLI::App app;
    c::configure(app, {.name = "probe"});
    int count = 12;
    auto* option = app.add_option("--count", count)->check(CLI::Range(1, 20))->capture_default_str();
    c::configuration config(app, [](auto const&) -> std::optional<std::string> { return "invalid"; });
    config.bind(option, count, "count").environment("COUNT").attribute("state", "Count");
    auto result = explicit_cli ? parse(config, {"--count=7"}) : parse(config, {});
    ASSERT_EQ(result.requested, c::action::run);
    config.values().attributes("state", {{"Count", 3}});
    auto metadata = config.resolve();
    EXPECT_EQ(count, explicit_cli ? 7 : 3);
    EXPECT_EQ(metadata.find("count")->value.get<int>(), count);
    config.values().set("count", 30);
    EXPECT_THROW((void)config.resolve(), std::invalid_argument);
  }
}

TEST(CliConfiguration, HelpDoesNotOpenFilesReadEnvironmentOrResolveFallbacks)
{
  CLI::App app;
  c::configure(app, {.name = "probe"});
  std::string value;
  c::configuration config(app, [](auto const&) -> std::optional<std::string> {
    ADD_FAILURE();
    return "x";
  });
  config.bind(app.add_option("--H", value)->required(), value, "hamiltonian").environment("H");
  config.option_file();
  auto result = parse(config, {"--help", "--config=/definitely/missing"});
  EXPECT_EQ(result.requested, c::action::help);
  auto help = uni20::presentation::render_plain(config.result_report({.name = "probe"}, result));
  EXPECT_NE(help.find("environment H"), std::string::npos);
  EXPECT_NE(help.find("required after loading sources"), std::string::npos);
  EXPECT_THROW((void)config.load_option_file(), std::logic_error);
  EXPECT_THROW((void)config.resolve(), std::logic_error);
}

TEST(CliConfiguration, RequiredOptionIsCheckedAfterLoadingAttributes)
{
  CLI::App app;
  c::configure(app, {.name = "probe"});
  std::string h;
  c::configuration config(app);
  config.bind(app.add_option("--H", h)->required(), h, "hamiltonian").attribute("state", "H");
  ASSERT_EQ(parse(config, {}).requested, c::action::run);
  EXPECT_THROW((void)config.resolve(), std::invalid_argument);
  config.values().attributes("state", {{"H", "Hubbard"}});
  EXPECT_NO_THROW((void)config.resolve());
  EXPECT_EQ(h, "Hubbard");
}

TEST(CliConfiguration, DeferredRealTextAndExactHalfIntegerFallback)
{
  CLI::App app;
  c::configure(app, {.name = "probe"});
  std::string u = "1";
  uni20::half_int spin(0);
  unsigned count = 12;
  c::configuration config(app);
  config.bind(app.add_option("--U", u), u, "u").attribute("state", "U");
  config.bind(c::add_half_int_option(app, "--spin", spin), spin, "spin").attribute("state", "S");
  config.bind(c::add_count_option(app, "--count", count), count, "count").attribute("state", "N");
  ASSERT_EQ(parse(config, {}).requested, c::action::run);
  config.values().attributes("state", {{"U", "1.000000000000000001"}, {"S", "3/2"}, {"N", 7}});
  auto metadata = config.resolve();
  EXPECT_EQ(uni20::parse_real<long double>(u), uni20::parse_real<long double>("1.000000000000000001"));
  EXPECT_EQ(spin, uni20::half_int::parse("1.5"));
  EXPECT_EQ(count, 7U);
  config.values().set("count", -1);
  EXPECT_THROW((void)config.resolve(), std::invalid_argument);
}

TEST(CliConfiguration, FileUsesCli11ParserAndUnknownKeysFail)
{
  char pattern[] = "/tmp/uni20-config-XXXXXX";
  int fd = mkstemp(pattern);
  ASSERT_GE(fd, 0);
  close(fd);
  struct cleanup
  {
      char const* path;
      ~cleanup() { unlink(path); }
  } cleanup{pattern};
  {
    std::ofstream file(pattern);
    file << "count = 7\nlabel = \"\"\n";
  }
  CLI::App app;
  c::configure(app, {.name = "probe"});
  unsigned count = 1;
  std::string label = "default";
  c::configuration config(app);
  config.bind(c::add_count_option(app, "--count", count), count, "count").file("count").attribute("state", "N");
  config.bind(app.add_option("--label", label), label, "label").file("label");
  config.option_file();
  ASSERT_EQ(parse(config, {"--config", pattern}).requested, c::action::run);
  config.load_option_file();
  config.values().attributes("state", {{"N", 6}});
  auto metadata = config.resolve();
  EXPECT_EQ(count, 7U);
  EXPECT_EQ(label, "");
  EXPECT_EQ(metadata.find("count")->options.origin->source, uni20::configuration_source::option_file);
  {
    std::ofstream file(pattern);
    file << "unknown = 8\n";
  }
  EXPECT_THROW((void)config.load_option_file(), std::invalid_argument);
}
