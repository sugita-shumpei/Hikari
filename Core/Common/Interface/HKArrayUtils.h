#ifndef HK_CORE_COMMON_ARRAY_UTILS_H
#define HK_CORE_COMMON_ARRAY_UTILS_H
#include <HKBytesUtils.h>
#include <tuple>
#include <array>
namespace HK
{
	template<typename T, size_t N>
	constexpr auto ToArray(const T(&arr)[N]) -> std::array<T, N>{
		std::array<T, N> res = {};
		for (size_t i = 0; i < N; ++i) {
			res[i] = arr[i];
		}
		return res;
	}

	template< size_t N, typename T>
	constexpr auto ToSubArray(const T* ptr) -> std::array<T, N> {
		std::array<T, N> res = {};
		for (size_t i = 0; i < N; ++i) {
			res[i] = ptr[i];
		}
		return res;
	}

	template<typename T, size_t N>
	auto AppendArrayL(const std::array<T, N>& arr, const T& val) -> std::array<T, N + 1>
	{
		std::array<T, N + 1> res = {};
		for (size_t i = 0; i < N ; ++i)
		{
			res[i] = arr[i];
		}
		res[N] = val;
		return res;
	}

	template<typename T, size_t N>
	auto AppendArrayR(const std::array<T, N>& arr, const T& val) -> std::array<T, N + 1>
	{
		std::array<T, N + 1> res = {};
		for (size_t i = 0; i < N; ++i)
		{
			res[i+1] = arr[i];
		}
		res[0] = val;
		return res;
	}

	template<typename T, size_t ...Ns>
	constexpr auto ConcatArray(const std::array<T, Ns>& ...arrs) -> std::array<T, (Ns + ...)>
	{
		std::array<T, (Ns + ...)> res = {};
		std::size_t off = 0;
		using Swallow = int[];
		{
			(void)Swallow {
				(HK::Copy(res.data() + off, arrs.data(), arrs.size()), off += Ns,0)...
			};
		}
		return res;
	}

	template<size_t ...Ns, typename T>
	constexpr auto SplitArray(const std::array<T, (Ns + ...)>& arr) -> std::tuple< std::array<T, Ns>...>
	{
		size_t offset  = 0;
		size_t offset2 = 0;
		std::tuple<std::array<T, Ns>...> res = {
			ToSubArray<Ns,T>(arr.data()+ (offset2=offset,offset+=Ns,offset2))...
		};
		return res;
	}

}
#endif