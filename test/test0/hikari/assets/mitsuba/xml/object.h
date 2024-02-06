#pragma once
#include <tinyxml2.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <memory>
#include <variant>
namespace hikari {
  namespace assets {
    namespace mitsuba {
      enum class XMLObjectType {
        eBsdf,
        eEmitter,
        eFilm,
        eIntegrator,
        eMedium,
        ePhase,
        eRFilter,
        eSampler,
        eSensor,
        eShape,
        eSpectrum,
        eTexture,
        eTransform,
        eVolume,
        eCount
      };
      struct XMLObject;
      using  XMLBoolean  = bool;
      using  XMLInteger  = int ;
      using  XMLFloat    = float;
      using  XMLString   = std::string;
      using  XMLVector   = glm::vec3;
      struct XMLPoint      {
        XMLPoint() noexcept :value{} {}
        explicit XMLPoint(const XMLVector& value_)noexcept :value{ value_ } {}
        XMLVector value;
      };
      struct XMLRef        {
        XMLRef() noexcept :id{""} {}
        explicit XMLRef(const XMLString& id_)noexcept:id{id_}{}
        XMLString id;
      };

      enum class XMLPropertyType {
        eBoolean  ,
        eInteger  ,
        eFloat    ,
        eString   ,
        eVector   ,
        ePoint    ,
        eRef      ,
        eObject   ,
        eUnknown
      };

      struct     XMLProperty {
        using Type = XMLPropertyType;
         XMLProperty() noexcept;
        ~XMLProperty() noexcept;

        XMLProperty(const XMLProperty&) noexcept = default;
        XMLProperty& operator=(const XMLProperty&) noexcept = default;

        auto getType() const noexcept -> Type;

        void setValue(XMLBoolean  value) noexcept;
        void setValue(XMLInteger  value) noexcept;
        void setValue(XMLFloat    value) noexcept;
        void setValue(XMLString   value) noexcept;
        void setValue(XMLVector   value) noexcept;
        void setValue(XMLPoint    value) noexcept;
        void setValue(XMLRef      value) noexcept;
        void setValue(std::shared_ptr<XMLObject> value) noexcept;

        bool getValue(XMLBoolean& value) const noexcept;
        bool getValue(XMLInteger& value) const noexcept;
        bool getValue(XMLFloat&   value) const noexcept;
        bool getValue(XMLString&  value) const noexcept;
        bool getValue(XMLVector&  value) const noexcept;
        bool getValue(XMLPoint&   value) const noexcept;
        bool getValue(XMLRef&     value) const noexcept;
        bool getValue(std::shared_ptr<XMLObject>& value) const noexcept;

        auto getBoolean  () const noexcept->std::optional<XMLBoolean>;
        auto getInteger  () const noexcept->std::optional<XMLInteger>;
        auto getFloat    () const noexcept->std::optional<XMLFloat>;
        auto getString   () const noexcept->std::optional<XMLString>;
        auto getVector   () const noexcept->std::optional<XMLVector>;
        auto getPoint    () const noexcept->std::optional<XMLPoint>;
        auto getRef      () const noexcept->std::optional<XMLRef>;
        auto getObject   () const noexcept->std::shared_ptr<XMLObject>;
      private:
        Type m_type = Type::eUnknown;
        std::variant<
          XMLBoolean,
          XMLInteger,
          XMLFloat  ,
          XMLString ,
          XMLVector ,
          XMLPoint  ,
          XMLRef    ,
          std::shared_ptr<XMLObject>,
          std::monostate
        > m_value = {};
      };
      struct     XMLProperties {
        using Type = XMLPropertyType;
        XMLProperties() noexcept;
        ~XMLProperties() noexcept;

        XMLProperties(const XMLProperties&) noexcept = default;
        XMLProperties& operator=(const XMLProperties&) noexcept = default;

        bool getValueType(const std::string& name, Type& type) const noexcept;
        auto getValueType(const std::string& name) const noexcept -> std::optional<Type>;

        bool hasValue(const std::string& name) const;
        bool hasValue(const std::string& name, Type  type) const;

        bool getValue(const std::string& name, XMLProperty& prop) const;
        auto getValue(const std::string& name) const->std::optional<XMLProperty>;

        void setValue(const std::string& name, XMLBoolean  value) noexcept;
        void setValue(const std::string& name, XMLInteger  value) noexcept;
        void setValue(const std::string& name, XMLFloat    value) noexcept;
        void setValue(const std::string& name, XMLString   value) noexcept;
        void setValue(const std::string& name, XMLVector   value) noexcept;
        void setValue(const std::string& name, XMLPoint    value) noexcept;
        void setValue(const std::string& name, XMLRef      value) noexcept;
        void setValue(const std::string& name, std::shared_ptr<XMLObject> value) noexcept;
        void setValue(const std::string& name, const XMLProperty& value) noexcept;

        bool getValue(const std::string& name, XMLBoolean& value) const noexcept;
        bool getValue(const std::string& name, XMLInteger& value) const noexcept;
        bool getValue(const std::string& name, XMLFloat  & value) const noexcept;
        bool getValue(const std::string& name, XMLString & value) const noexcept;
        bool getValue(const std::string& name, XMLVector & value) const noexcept;
        bool getValue(const std::string& name, XMLPoint  & value) const noexcept;
        bool getValue(const std::string& name, XMLRef    & value) const noexcept;
        bool getValue(const std::string& name, std::shared_ptr<XMLObject>& value) const noexcept;

        auto getBoolean(const std::string& name) const noexcept->std::optional<XMLBoolean>;
        auto getInteger(const std::string& name) const noexcept->std::optional<XMLInteger>;
        auto getFloat(const std::string& name) const noexcept->std::optional<XMLFloat>;
        auto getRef(const std::string& name) const noexcept->std::optional<XMLRef>;
        auto getString(const std::string& name) const noexcept->std::optional<XMLString>;
        auto getVector(const std::string& name) const noexcept->std::optional<XMLVector>;
        auto getPoint(const std::string& name) const noexcept->std::optional<XMLPoint>;
        auto getObject(const std::string& name) const noexcept->std::shared_ptr<XMLObject>;

        void eraseValue(const std::string& name);
        auto toString() const->std::string;
        auto toStrings() const->std::vector<std::string>;
      private:
        std::unordered_map<std::string, Type>                       m_types            = {};
        std::unordered_map<std::string, XMLBoolean >                m_value_booleans   = {};
        std::unordered_map<std::string, XMLInteger >                m_value_integers   = {};
        std::unordered_map<std::string, XMLFloat   >                m_value_floats     = {};
        std::unordered_map<std::string, XMLString  >                m_value_strings    = {};
        std::unordered_map<std::string, XMLVector  >                m_value_vectors    = {};
        std::unordered_map<std::string, XMLPoint   >                m_value_points     = {};
        std::unordered_map<std::string, XMLRef     >                m_value_refs       = {};
        std::unordered_map<std::string, std::shared_ptr<XMLObject>> m_value_objects    = {};
      };
      struct     XMLObject {
        using Type = XMLObjectType;
        virtual ~XMLObject();

        virtual auto toStrings() const->std::vector<std::string>;
        auto toString() const->std::string;

        auto getObjectType() const -> Type;
        auto getObjectTypeString() const->std::string;
        auto getPluginType() const -> std::string;

        auto getProperties() const -> const XMLProperties&;
        auto getProperties()       ->       XMLProperties&;

        auto getNestRefCount() const noexcept->size_t;
        void setNestRefCount(size_t count) noexcept;
        auto getNestRef(size_t idx) const noexcept-> XMLRef;
        void setNestRef(size_t idx, const XMLRef& ref) noexcept;
        void addNestRef(const XMLRef& ref) noexcept;

        auto getNestObjects() const noexcept->std::vector<std::shared_ptr<XMLObject>>;
        auto getNestObjCount(Type objectType) const noexcept->size_t;
        void setNestObjCount(Type objectType, size_t count) noexcept;
        auto getNestObj(Type objectType, size_t idx) const noexcept-> std::shared_ptr<XMLObject>;
        void setNestObj(size_t idx, const std::shared_ptr<XMLObject>& object)noexcept;
        void addNestObj(const std::shared_ptr<XMLObject>& object)noexcept;
      protected:
        XMLObject(Type object_type, const std::string& plugin_type);
      private:
        Type                                     m_object_type = {};
        std::string                              m_plugin_type = "";
        XMLProperties                            m_properties  = {};
        std::vector<XMLRef>                      m_nest_refs   = {};
        std::vector<std::shared_ptr<XMLObject>>  m_nest_objs[static_cast<int>(XMLObjectType::eCount)] = {};
      };
      struct XMLReferableObject : public XMLObject {
        virtual ~XMLReferableObject() noexcept;

        auto getID() const->std::string;
      protected:
        XMLReferableObject(Type object_type, const std::string& plugin_type, const std::string& id);
      private:
        std::string                              m_id = "";
      };

      using  XMLObjectPtr = std::shared_ptr<XMLObject>;
    }
  }
}
