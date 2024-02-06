#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLRFilter : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLRFilter>;
        virtual ~XMLRFilter() noexcept;
      private:
        XMLRFilter(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLRFilterPtr = std::shared_ptr<XMLRFilter>;
    }
  }
}
