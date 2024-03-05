#include <hikari/core/common/timer.h>
#include <hikari/core/window/app.h>
#include <hikari/core/window/window.h>
#include <hikari/core/ui/manager.h>
#include <hikari/core/ui/app.h>
#include <hikari/core/graphics/vulkan/renderer.h>
#include <hikari/core/graphics/opengl/renderer.h>
#include <hikari/core/graphics/opengl/ui_renderer.h>

namespace hikari {
  namespace test {
    struct Test0Renderer : public core::GraphicsOpenGLRenderer {
      Test0Renderer(core::Window* window) : core::GraphicsOpenGLRenderer(window) {}
      virtual ~Test0Renderer() noexcept {}
      void updateCustom() override
      {
        renderMain();
        renderUI();
      }
    private:
      void renderMain() {
        auto main_context = getRenderContext();
        main_context->setCurrent();
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        auto size = getSurfaceSize();
        glViewport(0, 0, size.width, size.height);
      }
      void renderUI() {
        ui_manager->render();
      }
    public:
      core::UIManager* ui_manager = nullptr;
    };
    struct Test0App : hikari::core::UIApp {
      Test0App(int argc, const char* argv[]) noexcept : hikari::core::UIApp(argc, argv){}
      virtual ~Test0App() noexcept {}
      bool initWindow() override {
        try {
          auto desc = hikari::core::WindowDesc();
          desc.position = { 100,100 };
          desc.size     = { 800u,600u };
          desc.title    = "";
          desc.flags    = hikari::core::WindowFlagBits::eGraphicsOpenGL| hikari::core::WindowFlagBits::eVisible | hikari::core::WindowFlagBits::eResizable;
          auto window = new hikari::core::Window(this, desc);
          window->setRenderer(std::unique_ptr<hikari::core::WindowRenderer>(new hikari::test::Test0Renderer(window)));
          setWindow(window);
          return true;
        }
        catch (std::runtime_error& err) {
          std::cerr << err.what() << std::endl;
          return false;
        }
      }
      bool initCustom() override {
        if (!initUI()) {
          return false;
        }
        return true;
      }
      void freeCustom() override {
        // Managerの終了時処理
        freeUI();
      }
    protected:
      bool initUI() override {
        auto window = getWindow();
        // Managerの初期化
        auto renderer = new core::GraphicsOpenGLUIRenderer(window);
        auto manager = std::make_unique<core::UIManager>(renderer);
        manager->initialize();
        auto window_renderer = (Test0Renderer*)getWindow()->getRenderer();
        window_renderer->ui_manager = manager.get();
        setUIManager(std::move(manager));
        return true;
      }
      void update() override {
        WindowApp::update();
        updateUI();
      }
      void onSync() override {
        hikari::core::WindowApp::onSync();
        onSyncUI();
      }
    };
  }
}

int main(int argc, const char** argv) {
  return hikari::test::Test0App(argc,argv).run();
}
