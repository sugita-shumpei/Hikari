#pragma once
#include <hikari/core/data_types.h>
#include <array>
namespace hikari {
  struct Application {
    static constexpr size_t max_argc = 256;
    virtual ~Application() noexcept {}
    auto getArgv(U64 i) const -> const char*;
    auto getArgc() const->I32;
    I32  run() {
      try {
        if (initialize()) {
          mainLoop();
        }
      }
      catch (...) {
        return -1;
      }
      terminate();
      return 0;
    }
  protected:
    Application(I32 argc, const Char* argv[])noexcept
      :m_argc{ std::min<I32>(std::max<I32>(argc,0), max_argc) } {
      for (size_t i = 0; i < m_argc; ++i) {
        m_argv[i] = argv[i];
      }
    }
  protected:
    virtual bool initialize() { return true; }
    virtual void mainLoop  () { }
    // terminate処理は例外を投げてはいけない
    virtual void terminate() noexcept { }
  private:
    I32                          m_argc = {};
    std::array<const Char*, 256> m_argv = {};
  };
}
