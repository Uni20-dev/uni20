#include <iostream>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/async/linear_solve.hpp>
#include <vector>

int main()
{
  using namespace uni20;
  using namespace uni20::async;
  using Matrix = DenseMatrix<mpreal>;
  TbbScheduler scheduler{4};
  ScopedScheduler scope(&scheduler);
  std::vector<Async<Matrix>> results;
  std::cout << "Four independent solves share a scheduler, using 80, 128, 256 and 400 bits.\n"
            << "Consumers are scheduled before their inputs. Each provider call sets and restores\n"
            << "precision on its executing thread after the inputs become readable.\n";
  for (int bits : {80, 128, 256, 400})
  {
    auto p = Precision::bits(bits);
    Async<Matrix> a, b;
    results.push_back(linalg::solve(a, b));
    scheduler.schedule([](WriteBuffer<Matrix> a, WriteBuffer<Matrix> b, Precision p) static -> AsyncTask {
      Matrix av(1, 1, p), bv(1, 1, p);
      av[0, 0] = mpreal(3, p);
      bv[0, 0] = mpreal(1, p);
      co_await a = std::move(av);
      co_await b = std::move(bv);
    }(a.write(), b.write(), p));
  }
  for (auto const& result : results)
  {
    auto const& value = result.get_wait();
    std::cout << value.default_precision().bit_count() << " bits: 1/3 = " << value[0, 0] << '\n';
  }
}
