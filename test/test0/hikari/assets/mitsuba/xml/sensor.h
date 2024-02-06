#pragma once
#include "context.h"
#include "object.h"
#include "medium.h"
#include "sampler.h"
#include "film.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLSensor : public XMLObject {
        static auto create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLSensor>;
        virtual ~XMLSensor() noexcept;

        void setSampler(const std::shared_ptr<XMLSampler>& sampler);
        auto getSampler() const->std::shared_ptr<XMLSampler>;
        void setFilm(const std::shared_ptr<XMLFilm>& film);
        auto getFilm() const -> std::shared_ptr<XMLFilm>;
        void setMedium(const std::shared_ptr<XMLMedium>& medium);
        auto getMedium() const->std::shared_ptr<XMLMedium>;
      private:
        XMLSensor(const XMLContextPtr& context, const std::string& plugin_type) noexcept;
      private:
        std::weak_ptr<XMLContext> m_context;
      };
      using XMLSensorPtr = std::shared_ptr<XMLSensor>;
    }
  }
}
