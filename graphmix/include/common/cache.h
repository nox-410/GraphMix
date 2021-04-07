#pragma once

#include <mutex>
#include <list>
#include <unordered_map>

namespace cache {

enum class policy {
  LRU,
  LFU,
  LFUOpt,
};

/*
  Cache:
    Cache is the Base class of all cache Policy
    args:
      limit: the number of Data will not exceed limit
    Note : Cache operations(insert, lookup, count, size) are not thread-safe.
    Note : Data must be copyable and movable. (you might use shared_ptr)
*/
template <typename cache_key_t, class Data>
class Cache {
protected:
  size_t limit_;
public:
  /*
    limit: cache size limit
  */
  explicit Cache(size_t limit) : limit_(limit) {}
  virtual ~Cache() {}
  size_t getLimit() { return limit_; }
  //------------------------- cache policy virtual function ---------------------
  virtual size_t size() = 0;
  virtual int count(cache_key_t k) = 0;
  virtual void insert(cache_key_t, const Data &data) = 0;
  virtual void lookup(cache_key_t k, Data &data) = 0;
}; // class Cache

/*
  LFUOptCache:
  Similar to LFU Cache, but performs better under static workload
*/
template <typename cache_key_t, class Data>
class LFUOptCache final : public Cache<cache_key_t, Data> {
private:
  struct Block {
    Data data;
    cache_key_t key;
    int use;
    Block(Data &&data, cache_key_t key, int use) : data(std::move(data)), key(key), use(use) {}
    Block(const Data &data, cache_key_t key, int use) : data(data), key(key), use(use) {}
  };
  typedef std::list<Block> CountList;
  typedef typename std::list<Block>::iterator iterator;
  const static int kUseCntMax = 10;
  CountList clist[kUseCntMax];
  std::unordered_map<cache_key_t, Data> store_;
  std::unordered_map<cache_key_t, iterator> hash_;

  // helper function
  iterator _increase(iterator iter) {
    size_t use = iter->use;
    clist[use + 1].emplace_front(std::move(iter->data), iter->key, iter->use + 1);
    clist[use].erase(iter);
    return clist[use + 1].begin();
  }

  iterator _create(cache_key_t key, const Data &data) {
    clist[0].emplace_front(data, key, 0);
    return clist[0].begin();
  }

  void _evict() {
    for (int i = 0; i < kUseCntMax; i++) {
      if (!clist[i].empty()) {
        cache_key_t key = clist[i].back().key;
        hash_.erase(key);
        clist[i].pop_back();
        break;
      }
    }
  }

public:
  using Cache<cache_key_t, Data>::Cache;
  size_t size() {
    return store_.size() + hash_.size();
  }

  int count(cache_key_t k) {
    return store_.count(k) + hash_.count(k);
  }

  void insert(cache_key_t key, const Data &data) {
    if (store_.count(key)) {
      store_[key] = data;
      return;
    }
    auto iter = hash_.find(key);
    if (iter != hash_.end()) {
      iter->second->data = data;
    } else {
      if (size() == this->limit_) {
        if (hash_.size() > 0) _evict();
        else return;
      }
      hash_[key] = _create(key, data);
    }
  }

  void lookup(cache_key_t k, Data &data) {
    auto store_ptr = store_.find(k);
    if (store_ptr != store_.end()) {
      data = store_ptr->second;
      return;
    }
    auto iter = hash_.find(k);
    if (iter == hash_.end()) return;
    data = iter->second->data;
    if (iter->second->use + 1 < kUseCntMax) hash_[k] = _increase(iter->second);
    else {
      store_[k] = std::move(iter->second->data);
      clist[kUseCntMax - 1].erase(iter->second);
      hash_.erase(k);
    }
  }
}; // class LFUCache

/*
  LRUCache:
    use LRU policy
    Implemented with a double-linked list and a hash map
    O(1) insert, lookup
*/
template <typename cache_key_t, class Data>
class LRUCache final : public Cache<cache_key_t, Data> {
private:
  typedef typename std::list<std::pair<cache_key_t, Data>>::iterator iterator;
  std::unordered_map<cache_key_t, iterator> hash_;
  std::list<std::pair<cache_key_t, Data>> list_;
public:
  using Cache<cache_key_t, Data>::Cache;
  size_t size() {
    return hash_.size();
  }

  int count(cache_key_t k) {
    return hash_.count(k);
  }

  void insert(cache_key_t k, const Data &data) {
    if (hash_.count(k)) {
      iterator iter = hash_[k];
      list_.erase(iter);
    }
    list_.emplace_front(k, data);
    hash_[k] = list_.begin();
    // Evict the least resently used if exceeds
    if (hash_.size() > this->limit_) {
      auto kv_pair = list_.back();
      hash_.erase(kv_pair.first);
      list_.pop_back();
    }
  }

  void lookup(cache_key_t k, Data &data) {
    auto iter = hash_.find(k);
    if (iter == hash_.end()) return;
    iterator list_iterator = iter->second;
    data = std::move(list_iterator->second);
    // Move the recently used key-value to the front of the list
    list_.erase(list_iterator);
    list_.emplace_front(k, data);
    hash_[k] = list_.begin();
  }

}; // class LRUCache

/*
  LFUCache:
    use LFU policy
    Implemented with hashmap and a 2-D list ordered by frequency
    O(1) insert and lookup
*/
template <typename cache_key_t, class Data>
class LFUCache final : public Cache<cache_key_t, Data> {
private:
  struct CountList;
  typedef typename std::list<CountList>::iterator clist_iterator;
  struct Block {
    Data data;
    cache_key_t key;
    clist_iterator head;
    Block(Data &&data, cache_key_t key, clist_iterator head) : data(std::move(data)), key(key), head(head) {}
    Block(const Data &data, cache_key_t key, clist_iterator head) : data(data), key(key), head(head) {}
  };
  typedef typename std::list<Block>::iterator iterator;
  struct CountList {
    std::list<Block> list;
    size_t use;
  };
  std::list<CountList> list_;
  std::unordered_map<cache_key_t, iterator> hash_;

  // helper function
  iterator _increase(iterator iter) {
    iterator result;
    auto clist = iter->head;
    auto clist_nxt = ++iter->head;
    size_t use = clist->use + 1;
    if (clist_nxt != list_.end() && clist_nxt->use == use) {
      clist_nxt->list.emplace_front(std::move(iter->data), iter->key, clist_nxt);
      result = clist_nxt->list.begin();
    } else {
      CountList temp = { {}, use };
      auto clist_new = list_.emplace(clist_nxt, temp);
      clist_new->list.emplace_front(std::move(iter->data), iter->key, clist_new);
      result = clist_new->list.begin();
    }
    clist->list.erase(iter);
    if (clist->list.empty())
      list_.erase(clist);
    return result;
  }

  iterator _create(cache_key_t key, const Data &data) {
    if (list_.empty() || list_.begin()->use > 1) {
      list_.push_front({ std::list<Block>(), 1 });
    }
    list_.begin()->list.emplace_front(data, key, list_.begin());
    return list_.begin()->list.begin();
  }

  void _evict() {
    auto clist = list_.begin();
    auto key = clist->list.back().key;
    hash_.erase(key);
    clist->list.pop_back();
    if (clist->list.empty())
      list_.erase(clist);
  }
public:
  using Cache<cache_key_t, Data>::Cache;
  size_t size() {
    return hash_.size();
  }

  int count(cache_key_t k) {
    return hash_.count(k);
  }

  void insert(cache_key_t k, const Data &data) {
    auto iter = hash_.find(k);
    if (iter == hash_.end()) {
      if (hash_.size() == this->limit_)
        _evict();
      hash_[k] = _create(k, data);
    } else {
      iter->second->data = data;
      hash_[k] = _increase(iter->second);
    }
  }

  void lookup(cache_key_t k, Data &data) {
    auto iter = hash_.find(k);
    if (iter == hash_.end()) return;
    hash_[k] = _increase(iter->second);
    data = iter->second->data;
  }
}; // class LFUCache

} // namespacce cache
