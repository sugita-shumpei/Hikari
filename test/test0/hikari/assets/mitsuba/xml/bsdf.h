#pragma once
#include "context.h"
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLBsdf : public XMLReferableObject {
        static auto create(const XMLContextPtr& context,const std::string& plugin_type, const std::string& id = "") -> std::shared_ptr<XMLBsdf>;
        virtual ~XMLBsdf() noexcept;

        auto getContext() const->XMLContextPtr;
      private:
        XMLBsdf(const XMLContextPtr& context, const std::string& plugin_type, const std::string& id) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLBsdfPtr = std::shared_ptr<XMLBsdf>;
    }
  }
}
