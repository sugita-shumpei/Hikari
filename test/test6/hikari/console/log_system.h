#pragma once
#include <hikari/core/singleton.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
namespace hikari {
  struct LogSystemImpl {
     LogSystemImpl() noexcept;
    ~LogSystemImpl() noexcept;
    auto getCoreLogger() noexcept ->std::shared_ptr<spdlog::logger>&;
    auto getClientLogger() noexcept ->std::shared_ptr<spdlog::logger>&;
    bool initialize();
    bool isInitialized() const noexcept;
  private:
    bool m_is_initialized = false;
    std::shared_ptr<spdlog::logger> m_core_logger   = nullptr;
    std::shared_ptr<spdlog::logger> m_client_logger = nullptr;
  };
  using LogSystem = SingletonWithInit<LogSystemImpl>;
}


#define HK_CORE_TRACE(...) ::hikari::LogSystem::getInstance()->getCoreLogger()->trace(__VA_ARGS__)
#define HK_CORE_INFO(...)  ::hikari::LogSystem::getInstance()->getCoreLogger()->info(__VA_ARGS__)
#define HK_CORE_WARN(...)  ::hikari::LogSystem::getInstance()->getCoreLogger()->warn(__VA_ARGS__)
#define HK_CORE_ERROR(...) ::hikari::LogSystem::getInstance()->getCoreLogger()->error(__VA_ARGS__)
#define HK_CORE_FATAL(...) ::hikari::LogSystem::getInstance()->getCoreLogger()->critical(__VA_ARGS__)

#define HK_TRACE(...) ::hikari::LogSystem::getInstance()->getClientLogger()->trace(__VA_ARGS__)
#define HK_INFO(...)  ::hikari::LogSystem::getInstance()->getClientLogger()->info(__VA_ARGS__)
#define HK_WARN(...)  ::hikari::LogSystem::getInstance()->getClientLogger()->warn(__VA_ARGS__)
#define HK_ERROR(...) ::hikari::LogSystem::getInstance()->getClientLogger()->error(__VA_ARGS__)
#define HK_FATAL(...) ::hikari::LogSystem::getInstance()->getClientLogger()->critical(__VA_ARGS__)
