#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLTexture : public XMLReferableObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type, const std::string& ref_id = "") -> std::shared_ptr<XMLTexture>;
        virtual ~XMLTexture() noexcept;
      private:
        XMLTexture(const XMLContextPtr& context, const std::string& plugin_type, const std::string& ref_id) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using  XMLTexturePtr = std::shared_ptr<XMLTexture>;
    }
  }
}
