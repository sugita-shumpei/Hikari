#pragma once
#include <hikari/core/data_type.h>
#include <hikari/core/color.h>
#include <hikari/core/shape.h>
#include <hikari/core/utils.h>
#include <hikari/shape/mesh.h>
namespace hikari
{
    inline namespace shape
    {

        struct ShapeSphereObject : public ShapeObject
        {
            using base_type = ShapeObject;
            static inline constexpr const char *TypeString() { return "ShapeSphere"; }
            static inline Bool Convertible(const Str &str) noexcept
            {
                if (base_type::Convertible(str))
                {
                    return true;
                }
                return str == TypeString();
            }
            static auto create(const Str &name, const Vec3 &c = Vec3(0.0f), F32 r = 1.0f) -> std::shared_ptr<ShapeSphereObject> { return std::shared_ptr<ShapeSphereObject>(new ShapeSphereObject(name, c, r)); }
            virtual ~ShapeSphereObject() noexcept {}

            Str getTypeString() const noexcept override { return TypeString(); }
            Bool isConvertible(const Str &type) const noexcept override
            {
                if (Convertible(type))
                {
                    return true;
                }
                return type == TypeString();
            }

            auto getName() const -> Str override { return m_name; }
            void setName(const Str &name) { m_name = name; }

            void setCenter(const Vec3 &c) { m_center = c; }
            auto getCenter() const -> Vec3 { return m_center; }

            auto getRadius() const -> F32 { return m_radius; }
            void setRadius(F32 radius) { m_radius = radius; }

            void setFlipNormals(Bool flip_normals) { m_flip_normals = flip_normals; }
            auto getFlipNormals() const -> Bool { return m_flip_normals; }

            // ShapeObject を介して継承されました
            auto getPropertyNames() const -> std::vector<Str> override
            {
                return {"name", "center", "radius", "flip_normals"};
            }
            void getPropertyBlock(PropertyBlockBase<Object> &pb) const override
            {
            }
            void setPropertyBlock(const PropertyBlockBase<Object> &pb) override
            {
            }
            Bool hasProperty(const Str &name) const override
            {
                if (name == "name")
                {
                    return true;
                }
                if (name == "center")
                {
                    return true;
                }
                if (name == "radius")
                {
                    return true;
                }
                if (name == "flip_normals")
                {
                    return true;
                }
                return false;
            }
            Bool getProperty(const Str &name, PropertyBase<Object> &prop) const override
            {
                if (name == "name")
                {
                    prop.setValue(getName());
                    return true;
                }
                if (name == "center")
                {
                    prop.setValue(getCenter());
                    return true;
                }
                if (name == "radius")
                {
                    prop.setValue(getRadius());
                    return true;
                }
                if (name == "flip_normals")
                {
                    prop.setValue(getFlipNormals());
                    return true;
                }
                return false;
            }
            Bool setProperty(const Str &name, const PropertyBase<Object> &prop) override
            {
                if (name == "name")
                {
                    auto v = prop.getValue<Str>();
                    if (!v)
                    {
                        return false;
                    }
                    setName(*v);
                    return true;
                }
                if (name == "center")
                {
                    auto v = prop.getValue<Vec3>();
                    if (!v)
                    {
                        return false;
                    }
                    setCenter(*v);
                    return true;
                }
                if (name == "radius")
                {
                    auto v = prop.getValue<F32>();
                    if (!v)
                    {
                        return false;
                    }
                    setRadius(*v);
                    return true;
                }
                if (name == "flip_normals")
                {
                    auto v = prop.getValue<Bool>();
                    if (!v)
                    {
                        return false;
                    }
                    setFlipNormals(*v);
                    return true;
                }
                return false;
            }
            auto getBBox() const -> BBox3 override
            {
                return BBox3(m_center - 0.5f * m_radius, m_center + 0.5f * m_radius);
            }

        protected:
            ShapeSphereObject(const Str &name, const Vec3 &c, F32 r) noexcept : ShapeObject(), m_name{name}, m_center{c}, m_radius{r} {}

        private:
            Str m_name;
            Vec3 m_center = Vec3(0.0f);
            F32 m_radius = 1.0f;
            Bool m_flip_normals = false;
        };
        struct ShapeSphere : protected ShapeImpl<ShapeSphereObject>
        {
            using impl_type = ShapeImpl<ShapeSphereObject>;
            using type = ShapeObject;

            ShapeSphere() noexcept : impl_type() {}
            ShapeSphere(const Str& name, const Vec3& c = Vec3(0.0f), F32 r = 1.0f) :impl_type(ShapeSphereObject::create(name,c,r)){}
            ShapeSphere(nullptr_t) noexcept : impl_type(nullptr) {}
            ShapeSphere(const ShapeSphere &) = default;
            ShapeSphere &operator=(const ShapeSphere &) = default;
            ShapeSphere(const std::shared_ptr<ShapeSphereObject> &object) : impl_type(object) {}
            ShapeSphere &operator=(const std::shared_ptr<ShapeSphereObject> &obj)
            {
                setObject(obj);
                return *this;
            }
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getCenter, Vec3, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getRadius, F32, 0.0f);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getFlipNormals, Bool, false);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setCenter, Vec3);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setRadius, F32);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setFlipNormals, Bool);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setName, Str);
            HK_METHOD_OVERLOAD_COMPARE_OPERATORS(ShapeSphere);
            using impl_type::operator[];
            using impl_type::operator!;
            using impl_type::operator bool;
            using impl_type::getBBox;
            using impl_type::getName;
            using impl_type::getObject;
            using impl_type::getPropertyBlock;
            using impl_type::getPropertyNames;
            using impl_type::getValue;
            using impl_type::hasValue;
            using impl_type::isConvertible;
            using impl_type::setObject;
            using impl_type::setPropertyBlock;
            using impl_type::setValue;
        };
        struct ShapeSphereSerializer : public ObjectSerializer
        {
            virtual ~ShapeSphereSerializer() noexcept {}

            // ObjectSerializer を介して継承されました
            auto getTypeString() const noexcept -> Str override;
            auto eval(const std::shared_ptr<Object> &object) const -> Json override;
        };
        struct ShapeSphereDeserializer : public ObjectDeserializer
        {
            virtual ~ShapeSphereDeserializer() noexcept {}

            // ObjectDeserializer を介して継承されました
            auto getTypeString() const noexcept -> Str override;
            auto eval(const Json &json) const -> std::shared_ptr<Object> override;
        };
    }
}
