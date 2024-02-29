#pragma once
#include <hikari/core/application.h>
#include <array>
namespace hikari {
  struct ConsoleApplication: public Application {
    static constexpr size_t max_argc = 256;
    ConsoleApplication(I32 argc, const Char* argv[])noexcept :Application(argc,argv){ }
    virtual ~ConsoleApplication() noexcept {}
  protected:
    using Application::initialize;
    using Application::terminate;
    using Application::mainLoop;
  };
}
