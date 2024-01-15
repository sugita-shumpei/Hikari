#pragma once
#include <hikari/core/film.h>
#include <hikari/core/data_type.h>
namespace hikari {
  struct FilmHdr : public Film
  {
    static constexpr Uuid ID() { return Uuid::from_string("67112552-BE76-4D35-9B79-36E84DC72DA6").value(); }
    static auto create(U32 width = 768, U32 height = 576) -> std::shared_ptr<FilmHdr>;
    virtual ~FilmHdr();

    Uuid getID() const override;

    auto getWidth () const->U32 override;
    auto getHeight() const->U32 override;

    void setWidth (U32 width)override;
    void setHeight(U32 height)override;

    void setComponentFormat(FilmComponentFormat component_format);
    auto getComponentFormat() const -> FilmComponentFormat;

    void setPixelFormat(FilmPixelFormat pixel_format);
    auto getPixelFormat() const->FilmPixelFormat;
  protected:
    FilmHdr(U32 width, U32 height);
  private:
    FilmComponentFormat m_component_format;
    FilmPixelFormat     m_pixel_format;
    U32                 m_width;
    U32                 m_height;
  };
}
