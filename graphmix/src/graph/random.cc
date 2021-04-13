#include "graph/random.h"

#include "common/logging.h"

std::mutex RandomIndexSelecter::mtx;
size_t RandomIndexSelecter::global_counter;

RandomIndexSelecter::RandomIndexSelecter() {
  std::lock_guard<std::mutex> lock(RandomIndexSelecter::mtx);
  RandomIndexSelecter::global_counter++;
  engine_.seed(global_counter);
}

std::unordered_set<size_t>
RandomIndexSelecter::unique(size_t n, size_t N) {
  std::unordered_set<size_t> result;
  if (n > N)
    LF << "Error random sequence n > N " << n << ">" << N;
  if (n <= N / 2) {
    result.reserve(n);
    while (result.size() < n) {
      size_t nxt = engine_() % N;
      if (!result.count(nxt)) result.emplace(nxt);
    }
    return result;
  } else {
    result.reserve(N);
    for (size_t i = 0; i < N; i++) result.emplace(i);
    while (result.size() > n) {
      size_t nxt = engine_() % N;
      if (result.count(nxt)) result.erase(nxt);
    }
    return result;
  }
}
