#pragma once
#include <hikari/core/data_type.h>
#include <hikari/core/color.h>
#include <hikari/core/shape.h>
#include <hikari/core/utils.h>
#include <hikari/core/serializer.h>
#include <hikari/core/deserializer.h>
namespace hikari
{
    inline namespace shape
    {
        enum class MeshIndexFormat
        {
            eU16,
            eU32
        };

        struct ShapeMeshObject : public ShapeObject
        {
            using base_type = ShapeObject;
            static inline constexpr const char *TypeString() { return "ShapeMesh"; }
            static inline Bool Convertible(const Str &str) noexcept
            {
                if (base_type::Convertible(str))
                {
                    return true;
                }
                return str == TypeString();
            }
            static auto create(const Str &name) -> std::shared_ptr<ShapeMeshObject>;
            virtual ~ShapeMeshObject() noexcept {}

            auto getName() const -> Str override { return m_name; }
            void setName(const Str &name) { m_name = name; }

            Str getTypeString() const noexcept override;
            Bool isConvertible(const Str &type) const noexcept override;

            auto getPropertyNames() const -> std::vector<Str> override;

            void getPropertyBlock(PropertyBlockBase<Object> &pb) const override;
            void setPropertyBlock(const PropertyBlockBase<Object> &pb) override;

            Bool hasProperty(const Str &name) const override;
            Bool getProperty(const Str &name, PropertyBase<Object> &prop) const override;
            Bool setProperty(const Str &name, const PropertyBase<Object> &prop) override;

            auto getIndices() const -> Array<U32> { return m_indices; }
            auto getPositions() const -> Array<Vec3> { return m_positions; }
            auto getNormals() const -> Array<Vec3> { return m_normals; }
            auto getTangents() const -> Array<Vec4> { return m_tangents; }
            auto getColors() const -> Array<ColorRGBA> { return m_colors; }
            template <U32 idx, std::enable_if_t<idx <= 8, nullptr_t> = nullptr>
            auto getUV() const -> Array<Vec3> { return m_uvs[idx]; }
            auto getUV(U32 idx) const -> Array<Vec3>
            {
                if (idx >= 9)
                {
                    return {};
                }
                else
                {
                    return m_uvs[idx];
                }
            }
#define HK_SHAPE_MESH_GET_UV_DEFINE(IDX) \
    Array<Vec3> getUV##IDX() const { return getUV<IDX>(); }
            HK_SHAPE_MESH_GET_UV_DEFINE(0);
            HK_SHAPE_MESH_GET_UV_DEFINE(1);
            HK_SHAPE_MESH_GET_UV_DEFINE(2);
            HK_SHAPE_MESH_GET_UV_DEFINE(3);
            HK_SHAPE_MESH_GET_UV_DEFINE(4);
            HK_SHAPE_MESH_GET_UV_DEFINE(5);
            HK_SHAPE_MESH_GET_UV_DEFINE(6);
            HK_SHAPE_MESH_GET_UV_DEFINE(7);
            HK_SHAPE_MESH_GET_UV_DEFINE(8);
            auto getUV() const -> Array<Vec3> { return getUV<0>(); }

            void setIndices(const Array<U32> &indices)
            {
                if (m_immutable)
                {
                    return;
                }
                if (indices.size() % 3 != 0)
                {
                    return;
                }
                m_indices = indices;
                m_index_count = indices.size() / 3;
                recalculateBBox();
            }
            void setPositions(const Array<Vec3> &positions)
            {
                if (m_immutable)
                {
                    return;
                }
                if (m_vertex_count != positions.size())
                {
                    m_normals.clear();
                    m_tangents.clear();
                    m_colors.clear();
                    for (auto &uv : m_uvs)
                    {
                        uv.clear();
                    }
                    m_vertex_count = positions.size();
                }
                m_positions = positions;
                recalculateBBox();
            }
            void setNormals(const Array<Vec3> &normals)
            {
                if (m_immutable)
                {
                    return;
                }
                if (normals.size() != m_vertex_count)
                {
                    return;
                }
                m_normals = normals;
            }
            void setTangents(const Array<Vec4> &tangents)
            {
                if (m_immutable)
                {
                    return;
                }
                if (tangents.size() != m_vertex_count)
                {
                    return;
                }
                m_tangents = tangents;
            }
            void setColors(const Array<ColorRGBA> &colors)
            {
                if (m_immutable)
                {
                    return;
                }
                if (colors.size() != m_vertex_count)
                {
                    return;
                }
                m_colors = colors;
            }
            void setUV(U32 idx, const Array<Vec2> &uv)
            {
                if (m_immutable)
                {
                    return;
                }
                if (idx >= 9)
                {
                    return;
                }
                if (uv.size() != m_vertex_count)
                {
                    return;
                }
                m_uvs[idx] = transformArray(uv, [](const auto &v)
                                            { return Vec3(v, 0.0f); });
            }
            void setUV(U32 idx, const Array<Vec3> &uv)
            {
                if (m_immutable)
                {
                    return;
                }
                if (idx >= 9)
                {
                    return;
                }
                if (uv.size() != m_vertex_count)
                {
                    return;
                }
                m_uvs[idx] = uv;
            }
            template <U32 idx, std::enable_if_t<idx <= 8, nullptr_t> = nullptr>
            void setUV(const Array<Vec2> &uv)
            {
                if (m_immutable)
                {
                    return;
                }
                if (idx >= 9)
                {
                    return;
                }
                if (uv.size() != m_vertex_count)
                {
                    return;
                }
                m_uvs[idx] = transformArray(uv, [](const auto &v)
                                            { return Vec3(v, 0.0f); });
            }
            template <U32 idx, std::enable_if_t<idx <= 8, nullptr_t> = nullptr>
            void setUV(const Array<Vec3> &uv)
            {
                if (m_immutable)
                {
                    return;
                }
                if (idx >= 9)
                {
                    return;
                }
                if (uv.size() != m_vertex_count)
                {
                    return;
                }
                m_uvs[idx] = uv;
            }
#define HK_SHAPE_MESH_SET_UV_DEFINE(IDX)                       \
    void setUV##IDX(const Array<Vec2> &uv) { setUV<IDX>(uv); } \
    void setUV##IDX(const Array<Vec3> &uv) { setUV<IDX>(uv); }
            HK_SHAPE_MESH_SET_UV_DEFINE(0);
            HK_SHAPE_MESH_SET_UV_DEFINE(1);
            HK_SHAPE_MESH_SET_UV_DEFINE(2);
            HK_SHAPE_MESH_SET_UV_DEFINE(3);
            HK_SHAPE_MESH_SET_UV_DEFINE(4);
            HK_SHAPE_MESH_SET_UV_DEFINE(5);
            HK_SHAPE_MESH_SET_UV_DEFINE(6);
            HK_SHAPE_MESH_SET_UV_DEFINE(7);
            HK_SHAPE_MESH_SET_UV_DEFINE(8);
            void setUV(const Array<Vec2> &uv) { setUV0(uv); }
            void setUV(const Array<Vec3> &uv) { setUV0(uv); }

            Bool hasIndices() const { return m_index_count > 0; }
            Bool hasPositions() const { return !m_positions.empty(); }
            Bool hasNormals() const { return !m_normals.empty(); }
            Bool hasTangents() const { return !m_tangents.empty(); }
            Bool hasColors() const { return !m_colors.empty(); }
            Bool hasUV(U32 idx) const
            {
                if (idx >= 9)
                {
                    return false;
                }
                return !m_uvs[idx].empty();
            }
            template <U32 idx, std::enable_if_t<idx <= 8, nullptr_t> = nullptr>
            Bool hasUV() const { return !m_uvs[idx].empty(); }
#define HK_SHAPE_MESH_HAS_UV_DEFINE(IDX) \
    Bool hasUV##IDX() const { return hasUV<IDX>(); }
            HK_SHAPE_MESH_HAS_UV_DEFINE(0);
            HK_SHAPE_MESH_HAS_UV_DEFINE(1);
            HK_SHAPE_MESH_HAS_UV_DEFINE(2);
            HK_SHAPE_MESH_HAS_UV_DEFINE(3);
            HK_SHAPE_MESH_HAS_UV_DEFINE(4);
            HK_SHAPE_MESH_HAS_UV_DEFINE(5);
            HK_SHAPE_MESH_HAS_UV_DEFINE(6);
            HK_SHAPE_MESH_HAS_UV_DEFINE(7);
            HK_SHAPE_MESH_HAS_UV_DEFINE(8);
            Bool hasUV() const { return hasUV0(); }

            auto getVertexCount() const -> U32 { return m_vertex_count; }
            auto getIndexCount() const -> U32 { return m_index_count; }

            auto getIndexFormat() const -> MeshIndexFormat { return m_index_format; }
            void setIndexFormat(MeshIndexFormat format) { m_index_format = format; }

            Bool isImmutable() const noexcept { return m_immutable; }
            void toImmutable() noexcept
            {
                if (m_immutable)
                {
                    return;
                }
                m_immutable = true;
            }

            Bool getFlipNormals() const noexcept { return m_flip_normals; }
            void setFlipNormals(Bool v) noexcept { m_flip_normals = v; }

            auto getBBox() const -> BBox3 override;
            void recalculateBBox();

        private:
            bool impl_getIndices(Property &prop) const;
            bool impl_getPositions(Property &prop) const;
            bool impl_getNormals(Property &prop) const;
            bool impl_getTangents(Property &prop) const;
            bool impl_getColors(Property &prop) const;
            bool impl_getUV(Property &prop) const;
            bool impl_getUV_without_idx_check(U32 idx, Property &prop) const;
            bool impl_getUV(U32 idx, Property &prop) const;

            bool impl_setIndices(const Property &prop);
            bool impl_setPositions(const Property &prop);
            bool impl_setNormals(const Property &prop);
            bool impl_setTangents(const Property &prop);
            bool impl_setColors(const Property &prop);
            bool impl_setUV(const Property &prop);
            bool impl_setUV_without_idx_check(U32 idx, const Property &prop);
            bool impl_setUV(U32 idx, const Property &prop);

        protected:
            ShapeMeshObject(const Str &name) noexcept : ShapeObject(), m_name{name} {}

        private:
            Array<U32> m_indices = {};
            Array<Vec3> m_positions = {};
            Array<Vec3> m_normals = {};
            Array<Vec4> m_tangents = {};
            Array<ColorRGBA> m_colors = {};
            Array<Vec3> m_uvs[9] = {};
            BBox3 m_bbox = {};
            MeshIndexFormat m_index_format = MeshIndexFormat::eU32;
            U32 m_vertex_count = 0;
            U32 m_index_count = 0;
            Bool m_immutable = false;
            Bool m_flip_normals = false;
            Str m_name = "";
        };
        template <typename ShapeMeshObjectT>
        struct ShapeMeshImpl : protected ShapeImpl<ShapeMeshObjectT>
        {
            using impl_type = ShapeImpl<ShapeMeshObjectT>;
            using type = typename impl_type::type;
            ShapeMeshImpl() noexcept : impl_type() {}
            ShapeMeshImpl(const Str &name) noexcept : impl_type(ShapeMeshObjectT::create(name)) {}
            ShapeMeshImpl(nullptr_t) noexcept : impl_type(nullptr) {}
            ShapeMeshImpl(const std::shared_ptr<ShapeMeshObjectT> &object) : impl_type(object) {}

            HK_METHOD_OVERLOAD_SETTER_LIKE(setName, Str);

            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getIndices, Array<U32>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getPositions, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getNormals, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getTangents, Array<Vec4>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getColors, Array<ColorRGBA>, {});
            auto getUV(U32 idx) const -> Array<Vec3>
            {
                auto obj = getObject();
                if (!obj)
                {
                    return {};
                }
                return obj->getUV(idx);
            }
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV0, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV1, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV2, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV3, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV4, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV5, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV6, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV7, Array<Vec3>, {});
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getUV8, Array<Vec3>, {});

            HK_METHOD_OVERLOAD_SETTER_LIKE(setIndices, Array<U32>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setPositions, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setNormals, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setTangents, Array<Vec4>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setColors, Array<ColorRGBA>);

            void setUV(U32 idx, const Array<Vec2> &uv)
            {
                auto obj = getObject();
                if (!obj)
                {
                    return;
                }
                obj->setUV(idx, uv);
            }
            void setUV(U32 idx, const Array<Vec3> &uv)
            {
                auto obj = getObject();
                if (!obj)
                {
                    return;
                }
                obj->setUV(idx, uv);
            }

            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV0, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV1, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV2, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV3, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV4, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV5, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV6, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV7, Array<Vec2>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV8, Array<Vec2>);

            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV0, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV1, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV2, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV3, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV4, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV5, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV6, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV7, Array<Vec3>);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setUV8, Array<Vec3>);

            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV0, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV1, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV2, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV3, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV4, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV5, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV6, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV7, Bool, false);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(hasUV8, Bool, false);

            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getVertexCount, U32, 0);
            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getIndexCount, U32, 0);

            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getIndexFormat, MeshIndexFormat, MeshIndexFormat::eU32);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setIndexFormat, MeshIndexFormat);

            HK_METHOD_OVERLOAD_GETTER_LIKE(isImmutable, Bool, false);
            void toImmutable() noexcept
            {
                auto obj = getObject();
                if (obj)
                {
                    obj->toImmutable();
                };
            }

            HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getFlipNormals, Bool, false);
            HK_METHOD_OVERLOAD_SETTER_LIKE(setFlipNormals, Bool);

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
        struct ShapeMesh : protected ShapeMeshImpl<ShapeMeshObject>
        {
            using impl_type = ShapeMeshImpl<ShapeMeshObject>;
            using type = typename impl_type::type;
            ShapeMesh() noexcept : impl_type() {}
            ShapeMesh(const Str &name) noexcept : impl_type(ShapeMeshObject::create(name)) {}
            ShapeMesh(nullptr_t) noexcept : impl_type(nullptr) {}
            ShapeMesh(const ShapeMesh &) = default;
            ShapeMesh &operator=(const ShapeMesh &) = default;
            ShapeMesh(const std::shared_ptr<ShapeMeshObject> &object) : impl_type(object) {}
            ShapeMesh &operator=(const std::shared_ptr<ShapeMeshObject> &obj)
            {
                setObject(obj);
                return *this;
            }
            HK_METHOD_OVERLOAD_COMPARE_OPERATORS(ShapeMesh);

            using impl_type::operator[];
            using impl_type::operator!;
            using impl_type::operator bool;
            using impl_type::getBBox;
            using impl_type::getColors;
            using impl_type::getFlipNormals;
            using impl_type::getIndexCount;
            using impl_type::getIndexFormat;
            using impl_type::getIndices;
            using impl_type::getName;
            using impl_type::getNormals;
            using impl_type::getObject;
            using impl_type::getPositions;
            using impl_type::getPropertyBlock;
            using impl_type::getPropertyNames;
            using impl_type::getTangents;
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
            using impl_type::getValue;
            using impl_type::getVertexCount;
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
            using impl_type::hasValue;
            using impl_type::isConvertible;
            using impl_type::isImmutable;
            using impl_type::setColors;
            using impl_type::setFlipNormals;
            using impl_type::setIndexFormat;
            using impl_type::setIndices;
            using impl_type::setNormals;
            using impl_type::setObject;
            using impl_type::setPositions;
            using impl_type::setPropertyBlock;
            using impl_type::setTangents;
            using impl_type::setUV;
            using impl_type::setUV0;
            using impl_type::setUV1;
            using impl_type::setUV2;
            using impl_type::setUV3;
            using impl_type::setUV4;
            using impl_type::setUV5;
            using impl_type::setUV6;
            using impl_type::setUV7;
            using impl_type::setUV8;
            using impl_type::setValue;
            using impl_type::toImmutable;
        };
        struct ShapeMeshSerializer : public ObjectSerializer
        {
            virtual ~ShapeMeshSerializer() noexcept {}

            // ObjectSerializer を介して継承されました
            auto getTypeString() const noexcept -> Str override;
            auto eval(const std::shared_ptr<Object> &object) const -> Json override;
        };
        struct ShapeMeshDeserializer : public ObjectDeserializer
        {
            virtual ~ShapeMeshDeserializer() noexcept {}

            // ObjectDeserializer を介して継承されました
            auto getTypeString() const noexcept -> Str override;
            auto eval(const Json &json) const -> std::shared_ptr<Object> override;
        };
    }
    inline namespace core {

      template <>
      struct EnumTraits<shape::MeshIndexFormat> : std::true_type
      {
        static auto toStr(const shape::MeshIndexFormat& t) -> Str
        {
          if (t == shape::MeshIndexFormat::eU16)
          {
            return "U16";
          }
          if (t == shape::MeshIndexFormat::eU32)
          {
            return "U32";
          }
          return "U32";
        }
        static auto toEnum(const Str& s) -> Option<shape::MeshIndexFormat>
        {
          if (s == "U16")
          {
            return shape::MeshIndexFormat::eU16;
          }
          if (s == "U32")
          {
            return shape::MeshIndexFormat::eU32;
          }
          return std::nullopt;
        }
      };
    }
}
