#pragma once

#define HK_METHOD_OVERLOAD_STATE_SHIFT_LIKE(METHOD) \
  void METHOD() { auto object = getObject(); if (object){ object->METHOD(); } }
#define HK_METHOD_OVERLOAD_SETTER_LIKE(METHOD, ARG) \
  void METHOD(const ARG& arg) { auto object = getObject(); if (object){ object->METHOD(arg); } }
#define HK_METHOD_OVERLOAD_GETTER_LIKE(METHOD, RES) \
  Option<RES> METHOD() const { auto object = getObject(); if (object){ return RES(object->METHOD()); } else { return std::nullopt;} }
#define HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(METHOD, RES,DEF) \
  RES METHOD() const { auto object = getObject(); if (object){ return RES(object->METHOD()); } else { return RES(DEF);} }
#define HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF_NO_CAST(METHOD, RES,DEF) \
  RES METHOD() const { auto object = getObject(); if (object){ return object->METHOD(); } else { return DEF;} }
#define HK_METHOD_OVERLOAD_GETTER_LIKE_OPTION(METHOD, RES) \
  Option<RES> METHOD() const { auto object = getObject(); if (object){ return object->METHOD(); } else { return std::nullopt;} }
#define HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(METHOD, RES) \
  Bool METHOD(RES& res) const { auto object = getObject(); if (object){ return object->METHOD(res); } return false; }
#define HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK_FROM_VOID(METHOD, RES) \
  Bool METHOD(RES& res) const { auto object = getObject(); if (object){ object->METHOD(res); return true; } return false; }
#define HK_METHOD_OVERLOAD_COMPARE_OPERATORS(TYPE) \
      Bool operator==(const TYPE& v)const { \
        auto obj1 = getObject(); \
        auto obj2 = v.getObject(); \
        return obj1 == obj2; \
      } \
      Bool operator!=(const TYPE& v)const { \
        auto obj1 = getObject(); \
        auto obj2 = v.getObject(); \
        return obj1 != obj2; \
      }
