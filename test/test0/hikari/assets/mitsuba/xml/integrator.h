#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLIntegrator : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLIntegrator>;
        virtual ~XMLIntegrator() noexcept;
      private:
        XMLIntegrator(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLIntegratorPtr = std::shared_ptr<XMLIntegrator>;
    }
  }
}
