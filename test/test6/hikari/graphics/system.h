#pragma once
#include <hikari/graphics/common.h>
namespace hikari {
  struct GraphicsInstance;
  struct GraphicsSystem  {
    static auto getInstance() -> GraphicsSystem&;
    ~GraphicsSystem() noexcept{}
    bool initGraphics(GraphicsAPIType api);
    void freeGraphics(GraphicsAPIType api);
    auto getGraphics(GraphicsAPIType api) -> GraphicsInstance*;
    template<typename GraphicsInstanceT>
    auto getGraphics() -> GraphicsInstanceT* {
      return static_cast<GraphicsInstanceT*>(getGraphics(GraphicsInstanceT::kAPIType));
    }
  private:
    GraphicsSystem() noexcept {}
    GraphicsSystem(const GraphicsSystem&) noexcept = delete;
    GraphicsSystem(GraphicsSystem&&) noexcept = delete;
    GraphicsSystem& operator=(const GraphicsSystem&) noexcept = delete;
    GraphicsSystem& operator=(GraphicsSystem&&) noexcept = delete;
  private:
    GraphicsInstance* m_instances[(size_t)GraphicsAPIType::eCount] = {};
  };
}
