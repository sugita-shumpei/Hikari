#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <type_traits>
#include <string_view>
#endif
#include <hikari/core/types/data_type.h>
#include <hikari/core/types/object_def.h>
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif
  
    // Object
    struct Object {
      static inline constexpr Bool Convertible(CStr type) noexcept { return std::string_view(type) == kTypeString; }
      static inline constexpr Char kTypeString[] = "Object";
      virtual Bool isConvertible(CStr type) const noexcept = 0;
      virtual CStr getTypeString()          const noexcept = 0;
      virtual ~Object() noexcept {}

      template<typename ObjectDeriveT, std::enable_if_t<std::is_base_of_v<Object,ObjectDeriveT> && !std::is_same_v<Object, ObjectDeriveT>,nullptr_t> = nullptr>
      Bool isConvertible() const noexcept { return isConvertible(ObjectDeriveT::kTypeString); }
      template<typename ObjectDeriveT, std::enable_if_t<std::is_same_v<Object, ObjectDeriveT>,nullptr_t> = nullptr>
      Bool isConvertible() const noexcept { return true; }
    };
    // ObjectUtils
    struct ObjectUtils {
      template<typename ObjectDeriveTo, typename ObjectDeriveFrom,
        std::enable_if_t<
        std::is_base_of_v<ObjectDeriveTo, ObjectDeriveFrom> &&
        std::is_base_of_v<Object, ObjectDeriveTo>   &&
        std::is_base_of_v<Object, ObjectDeriveFrom>
        , nullptr_t> = nullptr>
      static auto convert(const std::shared_ptr<ObjectDeriveFrom>& from) -> std::shared_ptr<ObjectDeriveTo> {
        return std::static_pointer_cast<ObjectDeriveTo>(from);
      }
      template<typename ObjectDeriveTo, typename ObjectDeriveFrom,
        std::enable_if_t<
       !std::is_base_of_v<ObjectDeriveTo, ObjectDeriveFrom>&&
        std::is_base_of_v<Object, ObjectDeriveTo>&&
        std::is_base_of_v<Object, ObjectDeriveFrom>
        , nullptr_t> = nullptr>
      static auto convert(const std::shared_ptr<ObjectDeriveFrom>& from) -> std::shared_ptr<ObjectDeriveTo> {
        if (!from) { return nullptr; }
        if (!from->isConvertible<ObjectDeriveTo>()) { return nullptr; }
        return std::static_pointer_cast<ObjectDeriveTo>(from);
      }
    };
    // ObjectHolder
    template<typename ObjectOwnerT, typename ObjectReturnT>
    struct SRefObjectHolder {
      SRefObjectHolder() noexcept : m_object{} {}
      SRefObjectHolder(const std::shared_ptr<ObjectOwnerT>& object) noexcept : m_object{ object } {}
      SRefObjectHolder(const SRefObjectHolder& lhs) noexcept : m_object{ lhs.m_object } {}
      auto getRef() const noexcept ->std::shared_ptr<ObjectReturnT> { return ObjectUtils::convert<ObjectReturnT>(m_object); }
      void setHolder(const SRefObjectHolder& c) noexcept { m_object = c.m_object; }
    private:
      std::shared_ptr<ObjectOwnerT> m_object;
    };
    template<typename ObjectOwnerT, typename ObjectReturnT>
    struct WRefObjectHolder {
      WRefObjectHolder() noexcept : m_object{}  {}
      WRefObjectHolder(const std::shared_ptr<ObjectOwnerT>& object) noexcept : m_object{ object } {}
      WRefObjectHolder(const WRefObjectHolder& lhs) noexcept : m_object{ lhs.m_object } {}
      auto getRef() const noexcept ->std::shared_ptr<ObjectReturnT> { return ObjectUtils::convert<ObjectReturnT>(m_object.lock()); }
      void setHolder(const WRefObjectHolder& c) noexcept { m_object = c.m_object; }
    private:
      std::weak_ptr<ObjectOwnerT> m_object;
    };
    template<typename ObjectOwnerT, typename ObjectReturnT>
    struct WRefObjectHolderChild {
      WRefObjectHolderChild() noexcept : m_object{}, m_index{ 0u } {}
      WRefObjectHolderChild(const std::shared_ptr<ObjectOwnerT>& object, U32 idx) noexcept : m_object{ object }, m_index{idx} {}
      WRefObjectHolderChild(const WRefObjectHolderChild& lhs) noexcept : m_object{ lhs.m_object }, m_index{ lhs.m_index } {}
      auto getRef() const noexcept ->std::shared_ptr<ObjectReturnT> {
        auto obj = m_object.lock();
        if (!obj) { return nullptr; }
        return ObjectUtils::convert<ObjectReturnT>(obj->getChild(m_index));
      }
      void setHolder(const WRefObjectHolderChild& c) noexcept { m_object = c.m_object; m_index = c.m_index; }
    private:
      std::weak_ptr<ObjectOwnerT> m_object;
      U32 m_index;
    };
    // RefObjectBase 
    template<typename ObjectOwnerT, typename ObjectReturnT, template<typename ObjectOwnerT, typename ObjectReturnT> typename ObjectHolder>
    struct RefObjectBase{
      using type        = ObjectReturnT;
      using holder_type = ObjectHolder<ObjectOwnerT, ObjectReturnT>;
        RefObjectBase() noexcept : m_holder{} {}
        RefObjectBase(nullptr_t) noexcept :m_holder{} {}
        RefObjectBase(const holder_type& holder) noexcept :m_holder{ holder } {}
       ~RefObjectBase() noexcept {}
       auto getRef() const ->std::shared_ptr<ObjectReturnT> { return m_holder.getRef(); }
       Bool isConvertible(CStr type) const noexcept {
         auto ref = getRef();
         if (!ref) { return true; }
         return ref->isConvertible(type);
       }
       CStr getTypeString() const noexcept {
         auto ref = getRef();
         if (!ref) { return ""; }
         return ref->getTypeString();
       }

       template<typename RefObjectDeriveT, std::enable_if_t<std::is_base_of_v<type, typename RefObjectDeriveT::type> && !std::is_same_v<type, typename RefObjectDeriveT::type>, nullptr_t> = nullptr>
       Bool isConvertible() const noexcept { return isConvertible(ObjectDeriveT::kTypeString); }
       template<typename RefObjectBaseT, std::enable_if_t<std::is_base_of_v<typename RefObjectBaseT::type, type>, nullptr_t> = nullptr>
       Bool isConvertible() const noexcept { return true; }

       template<typename RefObjectDeriveT, std::enable_if_t<std::is_base_of_v<type, typename RefObjectDeriveT::type> && !std::is_same_v<type, typename RefObjectDeriveT::type>, nullptr_t> = nullptr>
       auto convert() const noexcept -> RefObjectDeriveT { return RefObjectDeriveT(typename RefObjectDeriveT::holder_type(ObjectUtils::convert<typename RefObjectDeriveT::type>(getRef()))); }
       template<typename RefObjectBaseT, std::enable_if_t<std::is_base_of_v<typename RefObjectBaseT::type, type>, nullptr_t> = nullptr>
       auto convert() const noexcept -> RefObjectBaseT   { return RefObjectBaseT(typename RefObjectBaseT::holder_type(ObjectUtils::convert<typename RefObjectBaseT::type>(getRef()))); }
    protected:
      auto getHolder() const noexcept -> holder_type     { return m_holder;   }
      void setHolder(const holder_type& holder) noexcept { m_holder = holder; }
    private:
      holder_type m_holder;
    };
    // SRefObject
    struct SRefObject : protected RefObjectBase<Object, Object, SRefObjectHolder> {
      using impl_type = RefObjectBase<Object, Object, SRefObjectHolder>;
      HK_CORE_TYPES_REF_OBJECT_TYPE_ALIAS();
      SRefObject() noexcept : impl_type() {}
      HK_CORE_TYPES_REF_OBJECT_DEF_TO_BOOLEAN(SRefObject);
      HK_CORE_TYPES_REF_OBJECT_DEF_FROM_SHARED(SRefObject);
      HK_CORE_TYPES_REF_OBJECT_DEF_FROM_REFOBJ(SRefObject);
      HK_CORE_TYPES_REF_OBJECT_DEF_COPY_AND_MOVE(SRefObject);
      HK_CORE_TYPES_REF_OBJECT_DEF_COMPARISON(SRefObject);
      HK_CORE_TYPES_REF_OBJECT_USING_METHODS();
    };
    // WRefObject
    struct WRefObject : protected RefObjectBase<Object, Object, WRefObjectHolder> {
      using impl_type = RefObjectBase<Object, Object, WRefObjectHolder>;
      HK_CORE_TYPES_REF_OBJECT_TYPE_ALIAS();
      WRefObject() noexcept : impl_type() {}
      HK_CORE_TYPES_REF_OBJECT_DEF_FROM_SHARED(WRefObject);
      HK_CORE_TYPES_REF_OBJECT_DEF_FROM_REFOBJ(WRefObject);
      HK_CORE_TYPES_REF_OBJECT_DEF_COPY_AND_MOVE(WRefObject);
      HK_CORE_TYPES_REF_OBJECT_USING_METHODS();
    };
    // RefObjectUtils
    struct RefObjectUtils   {
      template<typename RefObjectDeriveTo, typename RefObjectDeriveFrom,
        std::enable_if_t<
        std::is_base_of_v<Object, typename RefObjectDeriveFrom::type>&&
        std::is_base_of_v<Object, typename RefObjectDeriveTo::type>
        , nullptr_t> = nullptr>
      static auto convert(const RefObjectDeriveFrom& from) -> RefObjectDeriveTo {
        return RefObjectDeriveTo(ObjectUtils::convert<typename RefObjectDeriveTo::type>(from.getRef()));
      }
    };

    typedef Array<SRefObject> ArraySRefObject;

#if defined(__cplusplus)
  }
}
#endif
