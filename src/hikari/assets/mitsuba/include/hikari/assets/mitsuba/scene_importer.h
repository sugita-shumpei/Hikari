#pragma once
#include <hikari/core/scene.h>
#include <memory>
namespace hikari {
  struct MitsubaSceneImporter {
    static auto create() ->    std::shared_ptr<MitsubaSceneImporter>;
    auto loadScene(const String& filename) -> std::shared_ptr<Scene>;
  private:
    MitsubaSceneImporter();
  private:
    struct          Impl;
    std::unique_ptr<Impl> m_impl;
  };
}
