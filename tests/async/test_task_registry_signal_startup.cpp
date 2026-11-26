#include <uni20/async/task_registry.hpp>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{

bool has_graphviz_file(std::filesystem::path const& directory, std::string const& prefix)
{
  std::error_code ec;
  for (auto const& entry : std::filesystem::directory_iterator(directory, ec))
  {
    if (ec) return false;
    auto const filename = entry.path().filename().string();
    if (filename.starts_with(prefix) && entry.path().extension() == ".dot") return true;
  }
  return false;
}

void remove_graphviz_files(std::filesystem::path const& directory, std::string const& prefix)
{
  std::error_code ec;
  for (auto const& entry : std::filesystem::directory_iterator(directory, ec))
  {
    if (ec) return;
    if (entry.path().filename().string().starts_with(prefix)) std::filesystem::remove(entry.path(), ec);
  }
}

bool child_skips_parent_service_shutdown()
{
  struct sigaction parent_action = {};
  if (::sigaction(SIGUSR1, nullptr, &parent_action) != 0) return false;

  auto const child = ::fork();
  if (child < 0) return false;
  if (child == 0)
  {
    ::alarm(5);
    uni20::detail::finalize_task_registry_diagnostics_from_environment();
    struct sigaction child_action = {};
    bool const preserved =
        ::sigaction(SIGUSR1, nullptr, &child_action) == 0 && child_action.sa_handler == parent_action.sa_handler;
    ::_exit(preserved ? 0 : 1);
  }

  int status{};
  while (::waitpid(child, &status, 0) < 0)
    if (errno != EINTR) return false;
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

} // namespace

int main()
{
  using namespace std::chrono_literals;

  auto const options = uni20::TaskRegistry::default_graphviz_dump_options();
  auto const output_dir = std::filesystem::path(options.output_dir);
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  remove_graphviz_files(output_dir, options.file_prefix);

  if (!child_skips_parent_service_shutdown())
  {
    std::fprintf(stderr, "child did not leave the parent's diagnostics service alone during shutdown\n");
    return 1;
  }

  if (::kill(::getpid(), SIGUSR1) != 0)
  {
    std::perror("kill(SIGUSR1)");
    return 1;
  }

  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline)
  {
    if (has_graphviz_file(output_dir, options.file_prefix))
    {
      uni20::TaskRegistry::stop_diagnostics_service();
      remove_graphviz_files(output_dir, options.file_prefix);
      return 0;
    }
    std::this_thread::sleep_for(10ms);
  }

  uni20::TaskRegistry::stop_diagnostics_service();
  remove_graphviz_files(output_dir, options.file_prefix);
  std::fprintf(stderr, "UNI20_DEBUG_DAG_SIGNAL did not start the diagnostics service before main\n");
  return 1;
}
