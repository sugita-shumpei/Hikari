#ifndef HK_CORE_COMMON_UUID_H
#define HK_CORE_COMMON_UUID_H
#include <HKMacro.h>
#include <HKArrayUtils.h>
#include <HKStringUtils.h>
#include <HKJSON.h>
#include <array>
#include <string>
struct  HKUUID
{
	using value_type = uint8_t;
	using reference = uint8_t&;
	using const_reference = const uint8_t&;
	using iterator = uint8_t*;
	using const_iterator = const uint8_t*;
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;

	constexpr HKUUID() noexcept :m_Data{} {}

	constexpr HKUUID(const HKUUID& lhs) noexcept = default;
	constexpr HKUUID(     HKUUID&& rhs) noexcept = default;
	constexpr HKUUID& operator=(const HKUUID& lhs) noexcept = default;
	constexpr HKUUID& operator=(HKUUID&& rhs) noexcept = default;

	constexpr HKUUID(const  std::array<uint8_t, 16>& arr)noexcept :m_Data{ arr } {}
	constexpr HKUUID(uint32_t v0, uint16_t v1, uint16_t v2, const std::array<uint8_t, 8>& v3)noexcept
	:m_Data{HK::ConcatArray(
			HK::ToBytes(std::array<uint32_t,1>{v0}),
			HK::ToBytes(std::array<uint16_t,1>{v1}),
			HK::ToBytes(std::array<uint16_t,1>{v2}),
			v3
	)}{}

	constexpr auto Size()const noexcept -> size_type { return 16; }

	constexpr iterator begin() noexcept { return m_Data.data(); }
	constexpr const_iterator begin() const noexcept { return m_Data.data(); }
	constexpr iterator end() noexcept { return m_Data.data()+Size(); }
	constexpr const_iterator end() const noexcept { return m_Data.data() + Size();}

	constexpr auto Data1()const noexcept -> uint32_t
	{
		return HK::FromBytes<uint32_t,1>(HK::ToSubArray<4>(m_Data.data() + 0))[0];
	}
	constexpr auto Data2()const noexcept -> uint16_t
	{
		return HK::FromBytes<uint16_t, 1>(HK::ToSubArray<2>(m_Data.data() + 4))[0];
	}
	constexpr auto Data3()const noexcept -> uint16_t
	{
		return HK::FromBytes<uint16_t, 1>(HK::ToSubArray<2>(m_Data.data() + 6))[0];
	}
	constexpr auto Data4()const noexcept -> std::array<uint8_t, 8> { return HK::ToSubArray<8>(m_Data.data() + 8); }

	constexpr static auto FromCStr(const char* str) -> HKUUID {
		auto data = HK::ConcatArray(
			HK::FromHexStrStorage<uint8_t, 4>(HK::ToSubArray< 8>(str + 0)),
			HK::FromHexStrStorage<uint8_t, 2>(HK::ToSubArray< 4>(str + 9)),
			HK::FromHexStrStorage<uint8_t, 2>(HK::ToSubArray< 4>(str + 14)),
			HK::FromHexStrStorage<uint8_t, 2>(HK::ToSubArray< 4>(str + 19)),
			HK::FromHexStrStorage<uint8_t, 6>(HK::ToSubArray<12>(str + 24))
		);
		return HKUUID(data);
	}

	constexpr auto ToBytes()const noexcept->const std::array<uint8_t, 16>& { return m_Data; }

	constexpr auto ToCStr()const noexcept -> std::array<char, 37>
	{
		auto v0 = HK::ToHexStrStorage(HK::ToSubArray<4>(m_Data.data()+ 0));
		auto v1 = HK::ToHexStrStorage(HK::ToSubArray<2>(m_Data.data()+ 4));
		auto v2 = HK::ToHexStrStorage(HK::ToSubArray<2>(m_Data.data()+ 6));
		auto v3 = HK::ToHexStrStorage(HK::ToSubArray<2>(m_Data.data()+ 8));
		auto v4 = HK::ToHexStrStorage(HK::ToSubArray<6>(m_Data.data()+ 10));
		return HK::ConcatArray(
			v0, std::array<char, 1>{'-'}, 
			v1, std::array<char, 1>{'-'},
			v2, std::array<char, 1>{'-'},
			v3, std::array<char, 1>{'-'},
			v4, std::array<char, 1>{'\0'}
		);
	}
	HK_CXX20_CONSTEXPR static auto FromString(std::string s) -> HKUUID {
		return FromCStr(s.data());
	}
	HK_CXX20_CONSTEXPR auto ToString()const noexcept -> std::string
	{
		auto str = ToCStr();
		return std::string(str.data());
	}
private:
	std::array<uint8_t, 16> m_Data;
};
#if __cplusplus >= 202002L
inline constexpr auto operator<=>(const HKUUID& rhs, const HKUUID& lhs)noexcept
{
	for (size_t i = 0; i < 15; ++i) {
		if (auto comp = (lhs.ToBytes()[i] <=> rhs.ToBytes()[i]); comp != 0) { return comp; }
	}
	return  (lhs.ToBytes()[15] <=> rhs.ToBytes()[15]);
}
#endif
inline constexpr auto operator== (const HKUUID& rhs, const HKUUID& lhs)noexcept
{
	for (size_t i = 0; i < 15; ++i) {
		if (auto comp = (lhs.ToBytes()[i] == rhs.ToBytes()[i]); comp != 0) { return comp; }
	}
	return  (lhs.ToBytes()[15] == rhs.ToBytes()[15]);
}
template<> struct std::hash<HKUUID>
{
	constexpr hash() = default;
	constexpr hash(const hash&) = default;
	constexpr hash(hash&&) = default;
	constexpr hash& operator=(const hash&) = default;
	constexpr hash& operator=(hash&&) = default;
	constexpr size_t operator()(HKUUID v)const {
		if constexpr (sizeof(size_t) == sizeof(uint64_t)) {
			return HK::FromBytes<uint64_t, 1>(std::array<uint8_t, 8>{
				v.ToBytes()[1], v.ToBytes()[3], v.ToBytes()[5], v.ToBytes()[7],
					v.ToBytes()[9], v.ToBytes()[11], v.ToBytes()[13], v.ToBytes()[15]
			})[0];
		}
		else
		{
			return HK::FromBytes<uint32_t, 1>(std::array<uint8_t, 4>{
				v.ToBytes()[3], v.ToBytes()[7], v.ToBytes()[11], v.ToBytes()[15],
			})[0];
		}
	}
};

namespace HK
{
	using UUID = HKUUID;
}
void from_json(const HKJSON& j, HKUUID& uuid)
{
	if (j.is_string())
	{
		auto s = j.get<std::string>();
		uuid = HKUUID::FromString(j);
	}
}

void to_json(HKJSON& j, const HKUUID& uuid)
{
	j = uuid.ToString();
}
#endif
