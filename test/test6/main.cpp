// #include <hikari/window/application.h>
#include <hikari/core/singleton.h>
#include <hikari/console/application.h>
#include <hikari/console/log_system.h>
#include <hikari/devices/system.h>
#include <hikari/events/system.h>
#include <hikari/events/event.h>
#include <hikari/events/queue.h>
#include <hikari/events/window_event.h>
#include <hikari/events/application_event.h>
#include <iostream>
#include <optional>
namespace hikari {
  struct SampleApp : public ConsoleApplication {
    SampleApp(I32 argc, const Char* argv[])noexcept :ConsoleApplication(argc, argv), m_main_window{ nullptr }, m_target{} {}
    virtual ~SampleApp() noexcept {}
    virtual bool initialize() override {
      auto& log = LogSystem::getInstance();
      if  (!log.initialize()) { return false; }
      auto& evt = EventSystem::getInstance();
      if (!evt.initialize()) { return false; }
      auto& dev = DeviceSystem::getInstance();
      if  (!dev.initialize()) { return false; }
      m_main_window = dev->createWindow(hikari::WindowCreateDesc::Builder().setWidth(800).setHeight(600).setPosX(100).setPosY(100).setTitle("tekitou").setFlags(hikari::WindowFlagBits::eVisible|hikari::WindowFlagBits::eResizable).build());
      // APP用のイベントTarget
      m_target = evt->createTarget("sample_app");
      m_target.addHandler(makeUniqueEventHandler<WindowCloseEvent>([this](const auto& e) { onCloseWindow(e.getWindow()); }));// Window がClose状態になったときに処理を実行
      m_target.addHandler(makeUniqueEventHandler<WindowDestroyEvent>([this](const auto& e) { onDestroyWindow(e.getWindow()); }));// WindowがDestroy状態になる前に必要な処理を実行
      m_target.addHandler(makeUniqueEventHandler<AppRemoveWindowEvent>([this](const auto& e) { onRemoveWindow(e.getWindow()); }));// AppがWindowを削除する際に使用(呼び出してはいけない)
      // LOG用のイベントTarget
      m_log_target = evt->createTarget("sample_app_log");
      m_log_target.addHandler(makeUniqueEventHandler<WindowResizeEvent>([this](const auto& e) { auto size = e.getSize(); HK_INFO("Event Window Resize: [{}, {}]!", size[0],size[1]); }));
      m_log_target.addHandler(makeUniqueEventHandler<WindowMovedEvent>([this](const auto& e) { auto pos = e.getPosition(); HK_INFO("Event Window Moved: [{}, {}]!", pos[0], pos[1]); }));
      m_log_target.addHandler(makeUniqueEventHandler<WindowCloseEvent>([this](const auto& e) { HK_INFO("Event Window Close!"); }));
      m_log_target.addHandler(makeUniqueEventHandler<WindowDestroyEvent>([this](const auto& e) { HK_INFO("Event Window Destroy!"); }));
      m_log_target.addHandler(makeUniqueEventHandler<AppRemoveWindowEvent>([this](const auto& e) { HK_INFO("Event App Remove Window!"); }));
      return true;
    }
    virtual void terminate() noexcept override {
      auto& dev = DeviceSystem::getInstance();
      auto& evt = EventSystem::getInstance();
      dev->destroyWindow(m_main_window);
      evt->destroyTarget(m_log_target );
      evt->destroyTarget(m_target     );
      dev.terminate();
      evt.terminate();
    }
    virtual void mainLoop() override {
      auto& dev = DeviceSystem::getInstance();
      auto& evt = EventSystem::getInstance();
      while (!shouldFinish()) {
        updateSystem();
        processEvent();
        update();
      }
    }
  private:
    bool shouldFinish() {
      return m_main_window == nullptr;
    }
    void updateSystem() {
      auto& dev = DeviceSystem::getInstance();
      auto& evt = EventSystem::getInstance();
      dev->update();// devicesystemを更新する(入出力の処理)       
      evt->update();// eventsystem を更新する(EventHandlerの更新)
    }
    void processEvent() {
      auto& evt = EventSystem::getInstance();
      evt->dispatchAll();// eventsをすべて処理する(基本的にはここでは必要な最小限度の処理のみ実行, Updateで実際の処理を起動)
    }
    // アプリケーションを更新する
    void update(){
      // dispatch window event
    }
  private:
    void onCloseWindow(Window* window) {
      auto& evt = EventSystem::getInstance();
      auto& que = evt->getGlobalQueue();
      que.push(std::make_unique<WindowDestroyEvent>(window));
    }
    void onDestroyWindow(Window* window) {
      auto& evt = EventSystem::getInstance();
      auto& que = evt->getGlobalQueue();
      // destroyはせず, アプリのdestroyを待機する
      que.push(std::make_unique<AppRemoveWindowEvent>(window));
    }
    void onRemoveWindow(Window* window) {
      auto& dev = DeviceSystem::getInstance();
      if (window == m_main_window) {
        m_main_window = nullptr;
      }
      dev->destroyWindow(window);
    }
  private:
    EventTarget m_target      = {};
    EventTarget m_log_target  = {};
    Window*     m_main_window = nullptr;
  };
}
int main(int argc, const char** argv) {
  return hikari::SampleApp(argc, argv).run();
}
