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
        struct ShapeRectObject : public ShapeMeshObject
        {
            using base_type = ShapeMeshObject;
            static inline constexpr const char *TypeString() { return "ShapeRect"; }
            static inline Bool Convertible(const Str &str) noexcept
            {
                if (base_type::Convertible(str))
                {
                    return true;
                }
                return str == TypeString();
            }
            static auto create(const Str &name) -> std::shared_ptr<ShapeRectObject>;
            virtual ~ShapeRectObject() noexcept {}

            Str getTypeString() const noexcept override { return TypeString(); }
            Bool isConvertible(const Str &type) const noexcept override
            {
                if (Convertible(type))
                {
                    return true;
                }
                return type == TypeString();
            }

        protected:
            ShapeRectObject(const Str &name) noexcept : ShapeMeshObject(name) {}
        };
        struct ShapeRect : protected ShapeMeshImpl<ShapeRectObject>
        {
            using impl_type = ShapeMeshImpl<ShapeRectObject>;
            using type = ShapeObject;

            ShapeRect() noexcept : impl_type() {}
            ShapeRect(const Str &name) : impl_type(ShapeRectObject::create(name)) {}
            ShapeRect(nullptr_t) noexcept : impl_type(nullptr) {}
            ShapeRect(const ShapeRect &) = default;
            ShapeRect &operator=(const ShapeRect &) = default;
            ShapeRect(const std::shared_ptr<ShapeRectObject> &object) : impl_type(object) {}
            ShapeRect &operator=(const std::shared_ptr<ShapeRectObject> &obj)
            {
                setObject(obj);
                return *this;
            }
            HK_METHOD_OVERLOAD_COMPARE_OPERATORS(ShapeRect);

            using impl_type::operator[];
            using impl_type::operator!;
            using impl_type::operator bool;
            using impl_type::getBBox;
            using impl_type::getFlipNormals;
            using impl_type::getName;
            using impl_type::getObject;
            using impl_type::getPropertyBlock;
            using impl_type::getPropertyNames;
            using impl_type::getValue;
            using impl_type::hasValue;
            using impl_type::isConvertible;
            using impl_type::isImmutable;
            using impl_type::setFlipNormals;
            using impl_type::setObject;
            using impl_type::setPropertyBlock;
            using impl_type::setValue;
            // using impl_type::toImmutable;
            using impl_type::getIndexCount;
            using impl_type::getIndexFormat;
            using impl_type::getVertexCount;
            // using impl_type::setIndexFormat;
            // using impl_type::setIndices;
            // using impl_type::setPositions;
            // using impl_type::setNormals;
            // using impl_type::setTangents;
            // using impl_type::setColors;
            using impl_type::getColors;
            using impl_type::getIndices;
            using impl_type::getNormals;
            using impl_type::getPositions;
            using impl_type::getTangents;
            // using impl_type::setUV;
            // using impl_type::setUV0;
            // using impl_type::setUV1;
            // using impl_type::setUV2;
            // using impl_type::setUV3;
            // using impl_type::setUV4;
            // using impl_type::setUV5;
            // using impl_type::setUV6;
            // using impl_type::setUV7;
            // using impl_type::setUV8;
            using impl_type::getUV;
            using impl_type::getUV0;
            using impl_type::getUV1;
            using impl_type::getUV2;
            using impl_type::getUV3;
            using impl_type::getUV4;
            using impl_type::getUV5;
            using impl_type::getUV6;
            using impl_type::getUV7;
            using impl_type::getUV8;
            using impl_type::hasUV;
            using impl_type::hasUV0;
            using impl_type::hasUV1;
            using impl_type::hasUV2;
            using impl_type::hasUV3;
            using impl_type::hasUV4;
            using impl_type::hasUV5;
            using impl_type::hasUV6;
            using impl_type::hasUV7;
            using impl_type::hasUV8;
        };
        struct ShapeRectSerializer : public ObjectSerializer
        {
            virtual ~ShapeRectSerializer() noexcept {}

            // ObjectSerializer を介して継承されました
            auto getTypeString() const noexcept -> Str override;
            auto eval(const std::shared_ptr<Object>& object) const->Json override;
        };
        struct ShapeRectDeserializer : public ObjectDeserializer
        {
            virtual ~ShapeRectDeserializer() noexcept {}

            // ObjectDeserializer を介して継承されました
            auto getTypeString() const noexcept -> Str override;
            auto eval(const Json& json) const->std::shared_ptr<Object> override;
        };
    }
}
