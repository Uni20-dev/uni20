#include "env_var_guard.hpp"
#include <gtest/gtest.h>
#include <uni20/common/display.hpp>
#include <uni20/common/program.hpp>

namespace p = uni20::presentation;

TEST(ProgramDocuments, IdentityAndBannerDoNotEvaluateReferenceProvider)
{
  unsigned calls = 0;
  p::program_info info{.name = "application",
                       .version = "2",
                       .revision = "checkout",
                       .authors = "Author",
                       .license = "License",
                       .references = [&] {
                         ++calls;
                         return std::vector<p::program_reference>{};
                       }};
  EXPECT_EQ(p::program_identity(info), "application 2 (checkout)");
  auto const text = p::render_plain(p::program_report(info));
  EXPECT_NE(text.find("Author"), std::string::npos);
  EXPECT_NE(text.find("License"), std::string::npos);
  EXPECT_EQ(calls, 0u);
  EXPECT_EQ(p::program_identity({.name = "unnumbered"}), "unnumbered");
}

TEST(ProgramDocuments, NarrowBannerWrapsProseThroughTheDisplaySink)
{
  p::program_info info{
      .name = "probe",
      .description = "Inspect Hubbard-style parameters using native precision; this example does not run a solver."};
  auto const report = p::program_report(info);
  std::string const expected = "probe\n"
                               "  Description  Inspect Hubbard-style\n"
                               "               parameters using native\n"
                               "               precision; this example\n"
                               "               does not run a solver.\n";
  auto policy = p::strict_ascii_policy();
  policy.wrap_width = 40;
  EXPECT_EQ(p::render_plain(report, policy), expected);

  uni20::test::EnvVarGuard columns("COLUMNS", "40");
  uni20::test::EnvVarGuard color("UNI20_COLOR", "never");
  uni20::display::scoped_sink use_default(uni20::display::sink{});
  ::testing::internal::CaptureStdout();
  uni20::display::emit(report, uni20::display::stream::out, false);
  EXPECT_EQ(::testing::internal::GetCapturedStdout(), expected);
}

TEST(ProgramDocuments, HelpUsesApplicationReferencesAndSemanticPolicies)
{
  p::program_info info{.name = "application", .references = [] {
                         return std::vector<p::program_reference>{
                             {.key = "model",
                              .citation = "Model citation",
                              .link = "https://example.org/model",
                              .applicability = "Relevant when selecting this model"}};
                       }};
  std::vector<p::help_group> groups{
      {.heading = "Model",
       .options = {{.names = "--spin HALF_INT", .description = "Exact spin", .attributes = {"default: 1/2"}}}}};
  auto const report = p::help_report(info, "application [options]", groups);
  auto const text = p::render_plain(report, p::strict_ascii_policy());
  for (auto const* expected : {"application [options]", "--spin HALF_INT", "default: 1/2", "Model citation",
                               "https://example.org/model", "Relevant when selecting this model"})
    EXPECT_NE(text.find(expected), std::string::npos);
  EXPECT_EQ(text.find("\033["), std::string::npos);
}

TEST(ProgramDocuments, NarrowHelpPreservesInvocationAndNumericTokens)
{
  std::string const number = "4.000000000000000000000000000001";
  std::string const default_number = "1.000000000000000000000000000002";
  std::string const command = "presentation_cli_example --U=" + number + " --precision=long-double";
  p::program_info info{.name = "probe", .examples = {{command, "Inspect a high-precision interaction"}}};
  std::vector<p::help_group> groups{{.heading = "Numerics",
                                     .options = {{.names = "--precision TEXT:{fp64,long-double,fp128}"},
                                                 {.names = "--U REAL", .attributes = {"default: " + default_number}}}}};
  auto const report = p::help_report(info, "probe [options]", groups);
  for (std::size_t width : {40u, 16u, 1u})
  {
    SCOPED_TRACE(width);
    auto policy = p::strict_ascii_policy();
    policy.wrap_width = width;
    auto const plain = p::render_plain(report, policy);
    policy.color = p::color_mode::always;
    auto const colored = p::render_terminal(report, policy);
    for (auto const& text : {plain, colored})
      for (auto const& token : {std::string("presentation_cli_example"), std::string("--precision=long-double"),
                                std::string("TEXT:{fp64,long-double,fp128}"), "--U=" + number, default_number})
        EXPECT_NE(text.find(token), std::string::npos) << token << '\n' << text;
  }
  // Exercise the actual default display sink, including its final layout handling.
  uni20::test::EnvVarGuard columns("COLUMNS", "40");
  uni20::test::EnvVarGuard color("UNI20_COLOR", "never");
  uni20::display::scoped_sink use_default(uni20::display::sink{});
  ::testing::internal::CaptureStdout();
  uni20::display::emit(report, uni20::display::stream::out);
  auto const output = ::testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("presentation_cli_example"), std::string::npos) << output;
  EXPECT_NE(output.find("--U=" + number), std::string::npos) << output;
  EXPECT_NE(output.find("--precision=long-double"), std::string::npos) << output;
  EXPECT_NE(output.find(default_number), std::string::npos) << output;
}
