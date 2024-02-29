#include <hikari/console/log_system.h>

hikari::LogSystemImpl::LogSystemImpl() noexcept {}

hikari::LogSystemImpl::~LogSystemImpl() noexcept {}

auto hikari::LogSystemImpl::getCoreLogger() noexcept -> std::shared_ptr<spdlog::logger>& {
  return m_core_logger;
}

auto hikari::LogSystemImpl::getClientLogger() noexcept -> std::shared_ptr<spdlog::logger>& {
  return m_client_logger;
}

bool hikari::LogSystemImpl::initialize() {
  if (m_is_initialized) { return true; }
  spdlog::set_pattern("%^[%T] %n: %v%$");
  m_core_logger = spdlog::stdout_color_mt("Hikari");
  m_core_logger->set_level(spdlog::level::trace);
  m_client_logger = spdlog::stdout_color_mt("App");
  m_client_logger->set_level(spdlog::level::trace);
  m_is_initialized = true;
  return true;
}

bool hikari::LogSystemImpl::isInitialized() const noexcept { return m_is_initialized; }
