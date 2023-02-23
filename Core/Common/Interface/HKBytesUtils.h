#ifndef HK_CORE_COMMON_BYTE_UTILS_H
#define HK_CORE_COMMON_BYTE_UTILS_H
#include <array>
namespace HK
{
	template<typename UIntT, bool Cond = std::is_unsigned_v<UIntT>>
	inline constexpr void Copy(UIntT* pSrc, const UIntT* pDst, size_t count) noexcept
	{
		for (size_t i = 0; i < count; ++i) {
			pSrc[i] = pDst[i];
		}
	}

	template<typename UIntT, size_t N, bool Cond = std::is_unsigned_v<UIntT>>
	inline constexpr auto FromBytes(const std::array<uint8_t, sizeof(UIntT)* N>& bytes) -> std::array<UIntT, N>
	{
		std::array<UIntT, N> res = {};
		for (size_t i = 0; i < N; ++i)
		{
			res[i] = 0;
			for (size_t j = 0; j < sizeof(UIntT); ++j)
			{
				res[i] += (static_cast<UIntT>(bytes[N * i + j]) << (8 * (sizeof(UIntT) - 1 - j)));
			}
		}
		return res;
	}
	template<typename UIntT, size_t N, bool Cond = std::is_unsigned_v<UIntT>>
	inline constexpr auto ToBytes(const std::array<UIntT, N>& arr) -> std::array<uint8_t, sizeof(UIntT)* N>
	{
		std::array<uint8_t, sizeof(UIntT)* N> res = {};
		for (size_t i = 0; i < N; ++i)
		{
			for (size_t j = 0; j < sizeof(UIntT); ++j)
			{
				res[sizeof(UIntT)*i+j] = static_cast<uint8_t>((arr[i]>>(8*(sizeof(UIntT)-1-j)))& static_cast<UIntT>(0xFF));
			}
		}
		return res;
	}
}
#endif
