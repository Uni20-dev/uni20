#include <uni20/common/program.hpp>
#ifdef TEST_CLI
#include <uni20/cli/cli.hpp>
#endif

int main()
{
  uni20::presentation::program_info info{.name = "consumer"};
  auto const report = uni20::presentation::program_report(info);
  if (uni20::presentation::render_plain(report) != "consumer\n") return 1;
#ifdef TEST_CLI
  CLI::App app;
  uni20::cli::configure(app, info);
  char const* argv[] = {"consumer", "--help"};
  if (uni20::cli::parse(app, 2, argv).requested != uni20::cli::action::help) return 2;
#endif
}
