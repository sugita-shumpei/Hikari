#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLEmitter : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLEmitter>;
        virtual ~XMLEmitter() noexcept;
      private:
        XMLEmitter(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLEmitterPtr = std::shared_ptr<XMLEmitter>;
    }
  }
}
