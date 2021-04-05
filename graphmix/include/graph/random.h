#include <random>
#include <mutex>
#include <unordered_set>

class RandomIndexSelecter {
public:
  RandomIndexSelecter();
  std::unordered_set<size_t> unique(size_t n, size_t N);
private:
  static size_t global_counter;
  static std::mutex mtx;
  std::mt19937_64 engine_;
};
