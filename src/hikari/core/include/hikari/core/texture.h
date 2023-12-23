#pragma once
#include <memory>
namespace hikari {
  enum class TextureFilterType {
    eBilinear,
    eNearest
  };
  enum class TextureWrapMode {
    eRepeat,
    eMirror,
    eClamp
  };
  struct Texture : public std::enable_shared_from_this<Texture> {
  public:
    virtual ~Texture() {}
    virtual Uuid getID() const = 0;
    template<typename DeriveType>
    auto convert() -> std::shared_ptr<DeriveType> {
      if (DeriveType::ID() == getID()) {
        return std::static_pointer_cast<DeriveType>(shared_from_this());
      }
      else {
        return nullptr;
      }
    }
  protected:
    Texture() {}
  };
  using TexturePtr = std::shared_ptr<Texture>;
}
