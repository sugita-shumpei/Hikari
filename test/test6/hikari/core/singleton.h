#pragma once
namespace hikari {
  template<typename Impl>
  struct Singleton {
    static auto getInstance() noexcept -> Singleton& {
      static Singleton instance;
      return instance;
    }
    ~Singleton() noexcept {}
    auto operator->() const noexcept -> const Impl* { return &m_impl; }
    auto operator->() noexcept -> Impl* { return &m_impl; }
  private:
    Singleton() noexcept : m_impl() {}
    Singleton(const Singleton&) noexcept = delete;
    Singleton& operator=(const Singleton&) noexcept = delete;
    Singleton(Singleton&&) noexcept = delete;
    Singleton& operator=(Singleton&&) noexcept = delete;
  private:
    Impl m_impl;
  };
  template<typename Impl>
  struct SingletonWithInit {
    static auto getInstance() noexcept -> SingletonWithInit& {
      static SingletonWithInit instance;
      return instance;
    }
    ~SingletonWithInit() noexcept {}
    bool operator! () const noexcept { return !isInitialized(); }
    operator bool () const noexcept { return  isInitialized(); }
    bool initialize()  { return m_impl.initialize(); }
    bool isInitialized() const noexcept { return m_impl.isInitialized(); }
    auto operator->() const noexcept -> const Impl* { return &m_impl; }
    auto operator->() noexcept -> Impl* { return &m_impl; }
  private:
    SingletonWithInit() noexcept : m_impl() {}
    SingletonWithInit(const SingletonWithInit&) noexcept = delete;
    SingletonWithInit& operator=(const SingletonWithInit&) noexcept = delete;
    SingletonWithInit(SingletonWithInit&&) noexcept = delete;
    SingletonWithInit& operator=(SingletonWithInit&&) noexcept = delete;
  private:
    Impl m_impl;
  };
  template<typename Impl>
  struct SingletonWithInitAndTerm {
    static auto getInstance() noexcept -> SingletonWithInitAndTerm& {
      static SingletonWithInitAndTerm instance;
      return instance;
    }
    ~SingletonWithInitAndTerm() noexcept {}
    bool operator! () const noexcept { return !isInitialized(); }
    operator bool() const noexcept { return  isInitialized(); }
    bool initialize() { return m_impl.initialize(); }
    void terminate() noexcept { return m_impl.terminate(); }
    bool isInitialized() const noexcept { return m_impl.isInitialized(); }
    auto operator->() const noexcept -> const Impl* { return &m_impl; }
    auto operator->() noexcept -> Impl* { return &m_impl; }
  private:
    SingletonWithInitAndTerm() noexcept : m_impl() {}
    SingletonWithInitAndTerm(const SingletonWithInitAndTerm&) noexcept = delete;
    SingletonWithInitAndTerm& operator=(const SingletonWithInitAndTerm&) noexcept = delete;
    SingletonWithInitAndTerm(SingletonWithInitAndTerm&&) noexcept = delete;
    SingletonWithInitAndTerm& operator=(SingletonWithInitAndTerm&&) noexcept = delete;
  private:
    Impl m_impl;
  };
}
