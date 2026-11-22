#include <uni20/async/task_registry.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include <signal.h>
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

} // namespace

int main(int argc, char** argv)
{
  using namespace std::chrono_literals;

  if (argc > 2) return 1;
  auto const options = uni20::TaskRegistry::default_graphviz_dump_options();
  auto const output_dir = std::filesystem::path(options.output_dir);
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  remove_graphviz_files(output_dir, options.file_prefix);

  // The file-trigger CTest supplies its own request path. The signal CTest always sends SIGUSR1.
  if (argc == 2)
  {
    uni20::TaskRegistry::DiagnosticsServiceOptions signal_options;
    signal_options.signal_number = SIGUSR1;
    if (uni20::TaskRegistry::start_diagnostics_service(signal_options))
    {
      std::fputs("a second start claimed the environment's file-only service\n", stderr);
      return 1;
    }

    std::filesystem::path const request_file(argv[1]);
    // The watch started with a relative path. Touch the original file after changing directory.
    std::filesystem::current_path(output_dir);
    if (std::filesystem::exists(request_file))
    {
      auto const previous = std::filesystem::last_write_time(request_file);
      std::filesystem::last_write_time(request_file, previous + 2s);
    }
    else if (!std::ofstream(request_file, std::ios::app))
    {
      std::perror(argv[1]);
      return 1;
    }
  }
  else if (::kill(::getpid(), SIGUSR1) != 0)
  {
    std::perror("kill(SIGUSR1)");
    return 1;
  }

  auto const deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline)
  {
    if (has_graphviz_file(output_dir, options.file_prefix)) return 0;
    std::this_thread::sleep_for(10ms);
  }

  std::fputs("the environment-started service did not produce the requested DOT file\n", stderr);
  return 1;
}
