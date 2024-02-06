#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLVolume : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLVolume>;
        virtual ~XMLVolume() noexcept;
      private:
        XMLVolume(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using  XMLVolumePtr = std::shared_ptr<XMLVolume>;
    }
  }
}
