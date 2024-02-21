#pragma once

#if defined(__cplusplus) && !defined(__CUDACC__)
#include <variant>
#endif

#include <hikari/core/types/property_def.h>
#include <hikari/core/types/object.h>
#include <hikari/core/types/data_type.h>
#include <hikari/core/types/vector.h>
#include <hikari/core/types/matrix.h>
#include <hikari/core/types/transform.h>
#include <hikari/core/types/color.h>
#include <hikari/core/types/utils.h>

#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif

    template<typename PropertyT, typename WRefPropertyHolder>
    struct WRefPropertyBase {
      WRefPropertyBase(const WRefPropertyHolder& holder) noexcept : m_holder{ holder } {}
      WRefPropertyBase(WRefPropertyBase&&) = delete;
      WRefPropertyBase(const WRefPropertyBase&) = delete;
      WRefPropertyBase& operator=(WRefPropertyBase&&) = delete;
      WRefPropertyBase& operator=(const WRefPropertyBase&) = delete;

      void operator=(nullptr_t& p) {
        m_holder.setProperty(PropertyT());
      }
      void operator=(const PropertyT& p) {
        m_holder.setProperty(p);
      }
      template<UPtr N>
       void operator=(const Char(&name)[N]) noexcept {
         m_holder.setProperty(Property(hikari::String(name)));
      }
      operator PropertyT() const {
        return m_holder.getProperty();
      }

      HK_WREF_PROPERTY_TYPE_DEFINE(Int);
      HK_WREF_PROPERTY_TYPE_DEFINE(Float);
      HK_WREF_PROPERTY_TYPE_DEFINE(UPtr);
      HK_WREF_PROPERTY_TYPE_DEFINE(IPtr);
      HK_WREF_PROPERTY_TYPE_DEFINE(Char);
      HK_WREF_PROPERTY_TYPE_DEFINE(Byte);
      HK_WREF_PROPERTY_TYPE_DEFINE(Bool);
      HK_WREF_PROPERTY_TYPE_DEFINE(String);
      HK_WREF_PROPERTY_TYPE_DEFINE(Vec2);
      HK_WREF_PROPERTY_TYPE_DEFINE(Vec3);
      HK_WREF_PROPERTY_TYPE_DEFINE(Vec4);
      HK_WREF_PROPERTY_TYPE_DEFINE(Mat2);
      HK_WREF_PROPERTY_TYPE_DEFINE(Mat3);
      HK_WREF_PROPERTY_TYPE_DEFINE(Mat4);
      HK_WREF_PROPERTY_TYPE_DEFINE(Quat);
      HK_WREF_PROPERTY_TYPE_DEFINE(Transform);
      HK_WREF_PROPERTY_TYPE_DEFINE(Color);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Int);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Float);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(UPtr);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(IPtr);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Char);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Byte);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Bool);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(String);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Vec2);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Vec3);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Vec4);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Mat2);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Mat3);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Mat4);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Quat);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Transform);
      HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(Color);

      HK_PROPERTY_VECTOR_AND_MATRIX_DEFINE();
      HK_PROPERTY_OBJECT_DEFINE();
      HK_WREF_PROPERTY_OBJECT_DEFINE_ASSIGN(WRefProperty);
    private:
      auto impl_getObject()const -> SRefObject {
        auto p= m_holder.getProperty();
        return p.getObject<SRefObject>();
      }
      void impl_setObject(const SRefObject& v) {
        m_holder.setProperty(Property(v));
      }
      auto impl_getArrayObject()const -> ArraySRefObject {
        auto p = m_holder.getProperty();
        return p.getArrayObject();
      }
      void impl_setArrayObject(const ArraySRefObject& v) {
        m_holder.setProperty(Property(v));
      }
    private:
      WRefPropertyHolder m_holder;
    };

    struct Property {
      Property() noexcept : m_data() {}
      Property(nullptr_t) noexcept : m_data() {}
      Property(std::monostate) noexcept : m_data() {}
      Property& operator=(nullptr_t) noexcept { m_data = std::monostate(); return *this; }
      Property& operator=(std::monostate) noexcept { m_data = std::monostate(); return *this; }
      Property(const Property& v) noexcept = default;
      Property& operator=(const Property& v) noexcept = default;
      Property(Property&& v) noexcept = default;
      Property& operator=(Property&& v) noexcept = default;
      template<UPtr N>
      Property(const Char(&name)[N]) noexcept :m_data{String(name)} {}
      template<UPtr N>
      Property& operator=(const Char(&name)[N]) noexcept {
        m_data = hikari::String(name);
        return *this;
      }

      Bool operator!() const noexcept { return !std::get_if<std::monostate>(&m_data); }
      operator Bool () const noexcept { return  std::get_if<std::monostate>(&m_data); }

      HK_PROPERTY_TYPE_DEFINE(Int);
      HK_PROPERTY_TYPE_DEFINE(Float);
      HK_PROPERTY_TYPE_DEFINE(UPtr);
      HK_PROPERTY_TYPE_DEFINE(IPtr);
      HK_PROPERTY_TYPE_DEFINE(Char);
      HK_PROPERTY_TYPE_DEFINE(Byte);
      HK_PROPERTY_TYPE_DEFINE(Bool);
      HK_PROPERTY_TYPE_DEFINE(String);
      HK_PROPERTY_TYPE_DEFINE(Vec2);
      HK_PROPERTY_TYPE_DEFINE(Vec3);
      HK_PROPERTY_TYPE_DEFINE(Vec4);
      HK_PROPERTY_TYPE_DEFINE(Mat2);
      HK_PROPERTY_TYPE_DEFINE(Mat3);
      HK_PROPERTY_TYPE_DEFINE(Mat4);
      HK_PROPERTY_TYPE_DEFINE(Quat);
      HK_PROPERTY_TYPE_DEFINE(Transform);
      HK_PROPERTY_TYPE_DEFINE(Color);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Int);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Float);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(UPtr);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(IPtr);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Char);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Byte);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Bool);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(String);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Vec2);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Vec3);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Vec4);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Mat2);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Mat3);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Mat4);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Quat);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Transform);
      HK_PROPERTY_ARRAY_TYPE_DEFINE(Color);
      HK_PROPERTY_VECTOR_AND_MATRIX_DEFINE();
      HK_PROPERTY_OBJECT_DEFINE();
      HK_PROPERTY_OBJECT_DEFINE_ASSIGN(Property);
    private:
      auto impl_getObject()const -> SRefObject {
        auto p = std::get_if<SRefObject>(&m_data);
        if (!p) { return SRefObject(); }
        return *p;
      }
      void impl_setObject(const SRefObject& v) {
        m_data = v;
      }
      auto impl_getArrayObject()const -> ArraySRefObject {
        auto p = std::get_if<ArraySRefObject>(&m_data);
        if (!p) { return ArraySRefObject(); }
        return *p;
      }
      void impl_setArrayObject(const ArraySRefObject& v) {
        m_data = v;
      }
    private:
      std::variant<
        Int   ,    // 整数               
        Float ,    // 浮動小数           
        UPtr  ,    // 符号無し64BIT整数: -> Address用(0xFFFF FFFF FFFF FFFFのように16進数)
        IPtr  ,    // 符号付き64BIT整数: -> 整数がInt領域を溢れた場合に使用
        Bool  ,    // Bool              
        Char  ,    // TextData          
        Byte  ,    // BinaryData(Base64)
        String,    // 文字列             
        Vec2  ,    // Vector型
        Vec3  ,
        Vec4  ,
        Mat2  ,    // Matrix型
        Mat3  ,
        Mat4  ,
        Quat  ,    // Quaternion型
        Transform, // Transform型
        Color,     // Color型
        SRefObject,// Object型
        ArrayInt,
        ArrayFloat,
        ArrayUPtr,
        ArrayIPtr,
        ArrayBool,
        ArrayChar,
        ArrayByte,
        ArrayString,
        ArrayVec2,
        ArrayVec3,
        ArrayVec4,
        ArrayMat2,
        ArrayMat3,
        ArrayMat4,
        ArrayQuat,
        ArrayTransform,
        ArrayColor,
        ArraySRefObject,
        std::monostate
      > m_data;
    };

    template<typename WRefPropertyHolder>
    using WRefProperty = WRefPropertyBase<Property, WRefPropertyHolder>;

#if defined(__cplusplus)
  }
}
#endif
