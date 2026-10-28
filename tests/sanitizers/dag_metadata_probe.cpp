#include <uni20/async/async_node.hpp>

#include <cstdlib>
#include <memory>
#include <new>
#include <string_view>

namespace
{
struct Payload
{
    inline static int live = 0;
    std::unique_ptr<int[]> data = std::make_unique<int[]>(128);
    Payload() { ++live; }
    ~Payload() { --live; }
};

// Intentional leaks are confined to subprocess negative controls.
[[gnu::noinline]] void evaluate(std::string_view mode)
{
  {
    auto storage = uni20::async::make_shared_storage<Payload>();
    if (mode == "metadata-raw" || mode == "metadata-storage" || mode == "owned")
    {
      auto* node = mode == "metadata-raw" ? uni20::async::NodeInfo::create(storage.get())
                                          : uni20::async::NodeInfo::create(storage);
      asm volatile("" : : "r"(node) : "memory");
    }
    if (mode == "owned")
    {
      // Destroy the wrapper/storage normally while abandoning its owned data.
      // No live diagnostic address points at this orphaned payload.
      auto* abandoned = storage.get()->data.release();
      asm volatile("" : : "r"(abandoned) : "memory");
    }
    if (mode == "plain")
    {
      void* abandoned = std::malloc(97);
      asm volatile("" : : "r"(abandoned) : "memory");
    }
  }
  if (Payload::live != 0) std::abort();
}
} // namespace

int main(int argc, char** argv)
{
  if (argc != 2) return 2;
  std::string_view mode(argv[1]);
  if (mode != "clean" && mode != "metadata-raw" && mode != "metadata-storage" && mode != "owned" && mode != "plain")
    return 2;
  evaluate(mode);
}
