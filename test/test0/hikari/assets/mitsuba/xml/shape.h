#pragma once
#include "object.h"
#include "bsdf.h"
#include "medium.h"
#include "transform.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLShape : public XMLReferableObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type, const std::string& id = "") -> std::shared_ptr<XMLShape>;
        virtual ~XMLShape() noexcept;
        // BSDF
        auto getBsdf() const noexcept -> std::shared_ptr<XMLBsdf>;
        void setBsdf(const std::shared_ptr<XMLBsdf>& bsdf) noexcept;
        // Medium
        auto getInteriorMedium() const noexcept -> std::shared_ptr<XMLMedium>;
        auto getExteriorMedium() const noexcept -> std::shared_ptr<XMLMedium>;
        void setInteriorMedium(const std::shared_ptr<XMLMedium>& medium)noexcept;
        void setExteriorMedium(const std::shared_ptr<XMLMedium>& medium)noexcept;
        // Context
        auto getContext() const->XMLContextPtr;
      private:
        XMLShape(const XMLContextPtr& context, const std::string& plugin_type, const std::string& id) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLShapePtr = std::shared_ptr<XMLShape>;
    }
  }
}
