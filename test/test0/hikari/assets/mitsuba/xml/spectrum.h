#pragma once
#include "context.h"
#include "object.h"
#include <string>
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLSpectrum : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLSpectrum>;
        virtual ~XMLSpectrum() noexcept;
      private:
        XMLSpectrum(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLSpectrumPtr = std::shared_ptr<XMLSpectrum>;
    }
  }
}
