#pragma once
#include "context.h"
#include "object.h"
#include "phase.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLMedium : public XMLReferableObject {
        static auto create(const XMLContextPtr& context,const std::string& plugin_type, const std::string& ref_id = "") -> std::shared_ptr<XMLMedium>;
        virtual ~XMLMedium() noexcept;

        auto getPhase() const->std::shared_ptr<XMLPhase>;
        void setPhase(const std::shared_ptr<XMLPhase>& phase);
      private:
        XMLMedium(const XMLContextPtr& context, const std::string& plugin_type, const std::string& ref_id) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLMediumPtr = std::shared_ptr<XMLMedium>;
    }
  }
}
