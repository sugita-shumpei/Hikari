#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLFilm : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLFilm>;
        virtual ~XMLFilm() noexcept;
      private:
        XMLFilm(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLFilmPtr = std::shared_ptr<XMLFilm>;
    }
  }
}
