#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <imath/half.h>
#include <nlohmann/json.hpp>
#include <hikari/core/tuple.h>

#define HK_TYPE_2_STRING_DEFINE(TYPE)          \
    template <>                                \
    struct Type2String<TYPE>                   \
    {                                          \
        static constexpr char value[] = #TYPE; \
    }

#define HK_TYPE_SINT_DEFINE(SIZE)    \
    typedef ::int##SIZE##_t I##SIZE; \
    HK_TYPE_2_STRING_DEFINE(I##SIZE)

#define HK_TYPE_UINT_DEFINE(SIZE)     \
    typedef ::uint##SIZE##_t U##SIZE; \
    HK_TYPE_2_STRING_DEFINE(U##SIZE)

#define HK_TYPE_VEC_DEFINE(SIZE)      \
    typedef glm::vec##SIZE Vec##SIZE; \
    HK_TYPE_2_STRING_DEFINE(Vec##SIZE)

#define HK_TYPE_MAT_DEFINE(SIZE)      \
    typedef glm::mat##SIZE Mat##SIZE; \
    HK_TYPE_2_STRING_DEFINE(Mat##SIZE)

namespace hikari
{
    inline namespace core
    {
        template <typename T>
        struct Type2String;

        template <typename TemplatTypeStrData, typename... Ts>
        struct TemplateTypeString : integer_sequence_to_str_data<
                                        typename concat_integer_sequence<
                                            typename str_data_to_integer_sequence<TemplatTypeStrData>::type,
                                            std::integer_sequence<char, '<'>,
                                            typename pop_integer_sequence_front<
                                                typename concat_integer_sequence<
                                                    typename concat_integer_sequence<
                                                        std::integer_sequence<char, ','>,
                                                        typename str_data_to_integer_sequence<Type2String<Ts>>::type>::type...>::type>::type,
                                            std::integer_sequence<char, '>'>>::type>
        {
        };

        template <typename T>
        using Option = std::optional<T>;
        namespace impl
        {
            struct OptionTypeString
            {
                static constexpr char value[] = "Option";
            };
        }

        template <typename T>
        struct Type2String<Option<T>> : TemplateTypeString<impl::OptionTypeString, T> {};

        template <typename T>
        using Array = std::vector<T>;
        namespace impl
        {
            struct ArrayTypeString
            {
                static constexpr char value[] = "Array";
            };
        }
        template <typename T, size_t IDX>
        using StaticArray = std::array<T, IDX>;

        template <>                                
        struct Type2String<std::monostate>
        {
          static constexpr char value[] = "None";
        };

        template <typename T>
        struct Type2String<Array<T>> : TemplateTypeString<impl::ArrayTypeString, T> {};

        template <typename Key, typename Val>
        using Dict = std::unordered_map<Key, Val>;
        namespace impl
        {
            struct DictTypeString
            {
                static constexpr char value[] = "Dict";
            };
        }

        template <typename Key, typename Val>
        struct Type2String<Dict<Key, Val>> : TemplateTypeString<impl::DictTypeString, Key, Val> {};

        template <typename Key, typename Val>
        using Pair = std::pair<Key, Val>;
        namespace impl
        {
          struct PairTypeString
          {
            static constexpr char value[] = "Pair";
          };
        }

        template <typename Key, typename Val>
        struct Type2String<Pair<Key,Val>> : TemplateTypeString<impl::PairTypeString, Key, Val> {};

        typedef Imath::half F16;
        typedef float F32;
        typedef double F64;
        typedef glm::quat Quat;
        typedef bool Bool;
        typedef unsigned char Byte;
        typedef char Char;
        typedef std::string Str;

        HK_TYPE_2_STRING_DEFINE(Str);
        HK_TYPE_2_STRING_DEFINE(Quat);
        HK_TYPE_2_STRING_DEFINE(Bool);

        HK_TYPE_2_STRING_DEFINE(F16);
        HK_TYPE_2_STRING_DEFINE(F32);
        HK_TYPE_2_STRING_DEFINE(F64);

        HK_TYPE_SINT_DEFINE(8);
        HK_TYPE_SINT_DEFINE(16);
        HK_TYPE_SINT_DEFINE(32);
        HK_TYPE_SINT_DEFINE(64);

        HK_TYPE_UINT_DEFINE(8);
        HK_TYPE_UINT_DEFINE(16);
        HK_TYPE_UINT_DEFINE(32);
        HK_TYPE_UINT_DEFINE(64);

        HK_TYPE_VEC_DEFINE(2);
        HK_TYPE_VEC_DEFINE(3);
        HK_TYPE_VEC_DEFINE(4);

        HK_TYPE_MAT_DEFINE(2);
        HK_TYPE_MAT_DEFINE(3);
        HK_TYPE_MAT_DEFINE(4);

        typedef nlohmann::json Json;
        HK_TYPE_2_STRING_DEFINE(Json);

        template <typename EnumT>
        struct EnumTraits : std::false_type {};

        template<typename EnumT, std::enable_if_t<EnumTraits<EnumT>::value, nullptr_t> = nullptr>
        inline auto convertEnum2Str(const EnumT& e) -> Str {
          return EnumTraits<EnumT>::toStr(e);
        }
        template<typename EnumT, std::enable_if_t<EnumTraits<EnumT>::value, nullptr_t> = nullptr>
        inline auto convertStr2Enum(const Str& s) -> Option<EnumT> {
          return EnumTraits<EnumT>::toEnum(s);
        }
    }
}

#undef HK_TYPE_SINT_DEFINE
#undef HK_TYPE_UINT_DEFINE
#undef HK_TYPE_VEC_DEFINE
#undef HK_TYPE_MAT_DEFINE
