#include <gtest/gtest.h>
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
