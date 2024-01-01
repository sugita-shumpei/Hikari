#pragma once
#include <hikari/core/scene.h>
#include <hikari/core/surface.h>
#include <memory>
namespace hikari {
  struct MitsubaSceneImporter {
    static auto create() ->    std::shared_ptr<MitsubaSceneImporter>;
    auto load(const String& filename) -> std::shared_ptr<Scene>;
    auto getSurfaceMap() const             -> const std::unordered_map<String, SurfacePtr>&;
  private:
    MitsubaSceneImporter();
  private:
    struct          Impl;
    std::unique_ptr<Impl> m_impl;
  };
}
