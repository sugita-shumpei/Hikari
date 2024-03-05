#pragma once
#include <hikari/core/window/app.h>
#include <hikari/core/ui/manager.h>
namespace hikari {
  namespace core {
    struct UIApp : public WindowApp {
    public:
      UIApp(int argc, const char* argv[]) noexcept:
        WindowApp(argc,argv){}
      virtual ~UIApp() noexcept{}
    protected:
      virtual bool initUI()
      {
        return true;
      }
      void freeUI() {
        m_ui_manager->terminate();
        m_ui_manager.reset();
      }
      void updateUI() {
        m_ui_manager->update();
      }
      void onSyncUI() {
        m_ui_manager->onSync();
      }
      void setUIManager(std::unique_ptr<UIManager>&& manager) { m_ui_manager = std::move(manager); }
      auto getUIManager() -> UIManager* { return m_ui_manager.get(); }
    private:
      std::unique_ptr<UIManager> m_ui_manager = nullptr;
    };
  }
}
