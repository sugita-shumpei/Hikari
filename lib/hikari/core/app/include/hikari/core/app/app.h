#pragma once
#include <exception>
namespace hikari {
  namespace core {
    struct App {
      App(int argc, const char* argv[]) noexcept;
      virtual ~App() noexcept;
      int run();
    private:
      virtual bool initialize() = 0;
      virtual void terminate() = 0;
      virtual void mainLoop() = 0;
    };
  }
}
