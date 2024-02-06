#pragma once
#include "object.h"
#include "context.h"
#include <glm/vec3.hpp>
namespace hikari {
  namespace assets {
    namespace mitsuba {
      enum class XMLTransformElementType {
        eTranslate,
        eRotate   ,
        eScale    ,
        eMatrix   ,
        eLookAt   ,
        eCount    
      };
      struct XMLTransformElement {
        using Type = XMLTransformElementType;
        virtual ~XMLTransformElement() {}
        virtual auto toString() const->std::string = 0;
        auto getType() const-> Type { return m_type; }
      protected:
        XMLTransformElement(Type type) :m_type{type} {}
      private:
        Type m_type;
      };
      using  XMLTransformElementPtr = std::shared_ptr<XMLTransformElement>;
      struct XMLTransform : public XMLObject {
        using Element    = XMLTransformElement;
        using ElementPtr = XMLTransformElementPtr;

        static auto create(const XMLContextPtr& context) -> std::shared_ptr<XMLTransform>;
        virtual ~XMLTransform() noexcept {}

        auto toStrings() const->std::vector<std::string> override;

        void addElement(const ElementPtr& elem) noexcept;
        void setElement(size_t idx, const ElementPtr& elem) noexcept;
        auto getElement(size_t idx) const noexcept -> ElementPtr;

        auto getSize() const noexcept ->size_t;
        void setSize(size_t idx) noexcept;
      private:
        XMLTransform(const XMLContextPtr& context) noexcept : XMLObject(Type::eTransform, ""), m_context{ context } {}
      private:
        std::weak_ptr<XMLContext> m_context = {};
        std::vector<std::shared_ptr<XMLTransformElement>> m_elements = {};
      };
      using  XMLTransformPtr = std::shared_ptr<XMLTransform>;
      struct XMLTransformElementTranslate : public XMLTransformElement {
        static auto create(const XMLContextPtr& context, const glm::vec3& position = {}) -> std::shared_ptr<XMLTransformElementTranslate>;
        virtual ~XMLTransformElementTranslate() noexcept;
        virtual auto toString() const->std::string override;
        void setValue(const glm::vec3& position) noexcept;
        auto getValue() const noexcept->glm::vec3;
      private:
        XMLTransformElementTranslate(const XMLContextPtr& context, const glm::vec3& position);
      private:
        std::weak_ptr<XMLContext> m_context = {};
        glm::vec3 m_value = {};
      };
      struct XMLTransformElementRotation : public XMLTransformElement {
        static auto create(const XMLContextPtr& context, const glm::vec3& value = {}, float angle = 0.0f) -> std::shared_ptr<XMLTransformElementRotation>;
        virtual ~XMLTransformElementRotation() noexcept;
        virtual auto toString() const->std::string override;
        void setValue(const glm::vec3& value) noexcept;
        auto getValue() const noexcept->glm::vec3;
        void setAngle(float angle) noexcept;
        auto getAngle() const noexcept -> float;
      private:
        XMLTransformElementRotation(const XMLContextPtr& context, const glm::vec3& value, float angle);
      private:
        std::weak_ptr<XMLContext> m_context = {};
        glm::vec3 m_value = {};
        float m_angle = 0.0f;
      };
      struct XMLTransformElementScale    : public XMLTransformElement {
        static auto create(const XMLContextPtr& context, const glm::vec3& value = glm::vec3(1.0f)) -> std::shared_ptr<XMLTransformElementScale>;
        virtual ~XMLTransformElementScale() noexcept;
        virtual auto toString() const->std::string override;
        void setValue(const glm::vec3& value) noexcept;
        auto getValue() const noexcept->glm::vec3;
      private:
        XMLTransformElementScale(const XMLContextPtr& context, const glm::vec3& value);
      private:
        std::weak_ptr<XMLContext> m_context = {};
        glm::vec3 m_value = {};
      };
      struct XMLTransformElementMatrix : public XMLTransformElement {
        static auto create(const XMLContextPtr& context, const glm::mat4& value = glm::mat4(1.0f)) -> std::shared_ptr<XMLTransformElementMatrix>;
        virtual ~XMLTransformElementMatrix() noexcept;
        virtual auto toString() const->std::string override;
        void setValue(const glm::mat4& value) noexcept;
        auto getValue() const noexcept->glm::mat4;
      private:
        XMLTransformElementMatrix(const XMLContextPtr& context, const glm::mat4& value);
      private:
        std::weak_ptr<XMLContext> m_context = {};
        glm::mat4 m_value;
      };
      struct XMLTransformElementLookAt : public XMLTransformElement {
        static auto create(const XMLContextPtr& context, const glm::vec3& origin = glm::vec3(0.0f), const glm::vec3& target = glm::vec3(0.0f,1.0f,0.0f), const glm::vec3& up = glm::vec3(0.0f,1.0f,0.0f)) -> std::shared_ptr<XMLTransformElementLookAt>;
        virtual ~XMLTransformElementLookAt() noexcept;
        virtual auto toString() const->std::string override;
        void setOrigin(const glm::vec3& origin) noexcept;
        void setTarget(const glm::vec3& target) noexcept;
        void setUp    (const glm::vec3& up    ) noexcept;
        auto getOrigin() const noexcept->glm::vec3;
        auto getTarget() const noexcept->glm::vec3;
        auto getUp    () const noexcept->glm::vec3;
      private:
        XMLTransformElementLookAt(const XMLContextPtr& context, const glm::vec3& origin,const glm::vec3& target, const glm::vec3& up);
      private:
        std::weak_ptr<XMLContext> m_context = {};
        glm::vec3 m_origin;
        glm::vec3 m_target;
        glm::vec3 m_up;
      };
    }
  }
}
