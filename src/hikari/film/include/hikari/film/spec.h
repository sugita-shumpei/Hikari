#pragma once
#include <hikari/core/film.h>
#include <hikari/core/data_type.h>
namespace hikari {
  struct FilmSpec : public Film
  {
    static constexpr Uuid ID() { return Uuid::from_string("877BE960-A9A6-4225-83C7-BDAB77B7894F").value(); }
    static auto create(U32 width = 768, U32 height = 576) -> std::shared_ptr<FilmSpec>;
    virtual ~FilmSpec();

    Uuid getID() const override;

    auto getWidth() const->U32 override;
    auto getHeight() const->U32 override;

    void setWidth(U32 width);
    void setHeight(U32 height);

    void setComponentFormat(FilmComponentFormat component_format);
    auto getComponentFormat() const->FilmComponentFormat;
  protected:
    FilmSpec(U32 width, U32 height);
  private:
    FilmComponentFormat m_component_format;
    U32                 m_width;
    U32                 m_height;
  };
}
