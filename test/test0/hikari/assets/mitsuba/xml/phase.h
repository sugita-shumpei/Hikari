#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLPhase : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLPhase>;
        virtual ~XMLPhase() noexcept;
      private:
        XMLPhase(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLPhasePtr = std::shared_ptr<XMLPhase>;
    }
  }
}
