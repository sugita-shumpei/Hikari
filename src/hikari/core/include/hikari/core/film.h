#pragma once
#include <hikari/core/data_type.h>
#include <memory>
namespace hikari {
  enum class FilmPixelFormat {
    eLuminance,
    eLuminanceAlpha,
    eRGB ,
    eRGBA,
    eXYZ ,
    eXYZA
  };
  enum class FilmComponentFormat {
    eFloat16,
    eFloat32,
    eUInt32  
  };

  struct Film : public std::enable_shared_from_this<Film> {
    virtual ~Film() {}
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

    virtual auto getWidth()  const->U32 = 0;
    virtual auto getHeight() const->U32 = 0;
    virtual void setWidth(U32 width) = 0;
    virtual void setHeight(U32 height) = 0;
  protected:
    Film() {}
  private:
  };
  using FilmPtr = std::shared_ptr<Film>;
}
