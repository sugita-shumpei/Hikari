#pragma once

#define HK_CORE_TYPES_OBJECT_DEFINE_TYPE_RELATIONSHIP(TYPE,BASE) \
  using type = TYPE; \
  using base_type = BASE; \
  static inline constexpr Char kTypeString[] = #TYPE; \
  static inline constexpr Bool Convertible(CStr type) noexcept { if (BASE::Convertible(type)){ return true;}return std::string_view(type) == kTypeString; }

#define HK_CORE_TYPES_OBJECT_IMPLEM_TYPE_RELATIONSHIP(TYPE,BASE) \
  HK_CORE_TYPES_OBJECT_DEFINE_TYPE_RELATIONSHIP(TYPE,BASE); \
  virtual Bool isConvertible(CStr type) const noexcept override { return Convertible(type); } \
  virtual CStr getTypeString()          const noexcept override { return getTypeString(); }

#define HK_CORE_TYPES_REF_OBJECT_BASE_DEF_CONS_AND_DEST(TYPE) \
    TYPE(nullptr_t) noexcept : impl_type(nullptr) {} \
    TYPE(const holder_type& holder) noexcept :impl_type(holder) {} \
    ~TYPE() noexcept {}

#define HK_CORE_TYPES_REF_OBJECT_TYPE_ALIAS() \
    using holder_type = typename impl_type::holder_type; \
    using type        = typename impl_type::type
#define HK_CORE_TYPES_REF_OBJECT_DEF_FROM_SHARED(TYPE) \
    TYPE(const std::shared_ptr<type>& lhs) noexcept : impl_type(holder_type(lhs)) {} \
    TYPE& operator=(const std::shared_ptr<type>& lhs) noexcept { setHolder(holder_type(lhs)); return *this; }
#define HK_CORE_TYPES_REF_OBJECT_DEF_FROM_REFOBJ(TYPE) \
    template<typename RefObjectDerived, std::enable_if_t<std::is_base_of_v<typename TYPE::type, typename RefObjectDerived::type> && !std::is_same_v<TYPE,RefObjectDerived>,nullptr_t> = nullptr> \
    TYPE(const RefObjectDerived& derived) : impl_type(holder_type(std::static_pointer_cast<typename TYPE::type>(derived.getRef()))){} \
    template<typename RefObjectDerived, std::enable_if_t<std::is_base_of_v<typename TYPE::type, typename RefObjectDerived::type> && !std::is_same_v<TYPE,RefObjectDerived>,nullptr_t> = nullptr> \
    TYPE& operator=(const RefObjectDerived& derived) noexcept { setHolder(holder_type(std::static_pointer_cast<typename TYPE::type>(derived.getRef()))); return *this; }

#define HK_CORE_TYPES_REF_OBJECT_DEF_COPY_AND_MOVE(TYPE) \
    TYPE(nullptr_t) noexcept : impl_type(nullptr) {} \
    TYPE(const TYPE& lhs) noexcept : impl_type(lhs.getHolder()) {} \
    ~TYPE() noexcept{} \
    TYPE(TYPE&& rhs) noexcept : impl_type(rhs.getHolder()) { rhs.setHolder(holder_type()); } \
    TYPE& operator=(nullptr_t) noexcept { \
      setHolder(holder_type()); \
      return *this; \
    } \
    TYPE& operator=(const TYPE& lhs) noexcept { \
      if (this != &lhs) { \
        setHolder(lhs.getHolder()); \
      } \
      return *this; \
    } \
    TYPE& operator=(TYPE&& rhs) noexcept { \
      if (this != &rhs) { \
        setHolder(rhs.getHolder()); \
        rhs.setHolder(holder_type()); \
      } \
      return *this; \
    }

#define HK_CORE_TYPES_REF_OBJECT_DEF_TO_BOOLEAN(TYPE) \
    Bool operator!() const noexcept { return getRef() == nullptr; } \
    operator Bool()  const noexcept { return getRef() != nullptr; }

#define HK_CORE_TYPES_REF_OBJECT_DEF_COMPARISON(TYPE) \
    Bool operator==(const TYPE& lhs) const noexcept { return getRef() == lhs.getRef(); } \
    Bool operator!=(const TYPE& lhs) const noexcept { return getRef() != lhs.getRef(); }

#define HK_CORE_TYPES_REF_OBJECT_USING_METHODS() \
    protected: \
      using impl_type::getHolder; \
      using impl_type::setHolder; \
    public: \
      using impl_type::getRef; \
      using impl_type::isConvertible; \
      using impl_type::getTypeString
