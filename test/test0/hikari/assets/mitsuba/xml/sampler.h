#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLSampler : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLSampler>;
        virtual ~XMLSampler() noexcept;
      private:
        XMLSampler(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using  XMLSamplerPtr = std::shared_ptr<XMLSampler>;
    }
  }
}
