#pragma once
#include "object.h"
#include "context.h"
#include "integrator.h"
#include "sensor.h"
#include "shape.h"
#include "emitter.h"
#include "transform.h"
#include "texture.h"
#include "spectrum.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLScene;
      struct XMLSceneImporter {
        static auto create(const std::string& filename) noexcept -> std::shared_ptr<XMLSceneImporter>;
        ~XMLSceneImporter() noexcept;

        void setFilename(const std::string& filename) { m_filename = filename; }
        auto getFilename() const->std::string { return m_filename; }

        auto loadScene() -> std::shared_ptr<XMLScene>;
      private:
        XMLSceneImporter(const std::string& filename) :m_filename{ filename } {}
        static auto loadXMLObject(std::shared_ptr<hikari::assets::mitsuba::XMLContext> context, void*) -> std::shared_ptr<XMLObject>;
        auto loadChildScene(const XMLContextPtr& context, const std::string& filename) -> std::shared_ptr<XMLScene>;
      private:
        std::string m_filename;
      };
      struct XMLScene {
        using Importer = XMLSceneImporter;
        static auto create() noexcept -> std::shared_ptr<XMLScene>;
        ~XMLScene() noexcept;
        auto toString()      const -> std::string;
        auto getContext()    const -> XMLContextPtr              { return m_own_context;  }
        auto getIntegrator() const -> XMLIntegratorPtr           { return m_integrator; }
        auto getSensor    () const -> XMLSensorPtr               { return m_sensor;}
        auto getShapes    () const -> std::vector<XMLShapePtr>   { return m_shapes;  }
        auto getEmitters  () const -> std::vector<XMLEmitterPtr> { return m_emitters;}
      private:
        friend class XMLSceneImporter;
        XMLScene();
        void setContext   (const XMLContextPtr   & context   ) { m_own_context = context;    }
        void setIntegrator(const XMLIntegratorPtr& integrator) { m_integrator  = integrator; }
        void setSensor    (const XMLSensorPtr    & sensor    ) { m_sensor      = sensor;     }
        void setShapes    (const std::vector<XMLShapePtr>&   shapes   ) { m_shapes = shapes; }
        void setEmitters  (const std::vector<XMLEmitterPtr>& emitters ) { m_emitters = emitters; }
      private:
        XMLContextPtr              m_own_context = nullptr;
        XMLIntegratorPtr           m_integrator  = nullptr;
        XMLSensorPtr               m_sensor      = nullptr;
        std::vector<XMLShapePtr>   m_shapes      = {};
        std::vector<XMLEmitterPtr> m_emitters    = {};
      };
      using  XMLScenePtr = std::shared_ptr<XMLScene>;
    }
  }
}
