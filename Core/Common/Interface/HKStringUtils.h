#ifndef HK_CORE_COMMON_CSTR_UTILS_H
#define HK_CORE_COMMON_CSTR_UTILS_H
namespace HK
{
	template<size_t N, typename T>
	constexpr auto ToSubStrStorage(const T* ptr) -> std::array<T, N+1>{
		std::array<T, N+1> res = {};
		for (size_t i = 0; i < N; ++i) {
			res[i] = ptr[i];
		}
		res[N] = '\0';
		return res;
	}

	template<typename T, size_t N>
	constexpr auto AppendStrStorageL(const std::array<T, N>& arr, const T& val) -> std::array<T, N + 1>
	{
		std::array<T, N + 1> res = {};
		for (size_t i = 0; i < N - 1; ++i)
		{
			res[i] = arr[i];
		}
		res[N-1] = val;
		res[N] = '\0';
		return res;
	}

	template<typename T, size_t N>
	constexpr auto AppendStrStorageR(const std::array<T, N>& arr, const T& val) -> std::array<T, N + 1>
	{
		std::array<T, N + 1> res = {};
		for (size_t i = 0; i < N - 1; ++i)
		{
			res[i+1] = arr[i];
		}
		res[0] = val;
		res[N] = '\0';
		return res;
	}

	template<typename T, size_t ...Ns>
	constexpr auto ConcatStrStorage(const std::array<T, Ns>& ...strs) -> std::array<T, ((Ns - 1) + ...) + 1>
	{
		std::array<T, ((Ns-1) + ...)+1> res = {};
		std::size_t off = 0;
		using Swallow = int[];
		{
			(void)Swallow {
				(HK::Copy(res.data() + off, strs.data(), Ns-1), off +=(Ns-1), 0)...
			};
		}
		res[off] = '\0';
		return res;
	}
	template<size_t ...Ns, typename T>
	constexpr auto SplitStrStorage(const std::array<T, (Ns + ...)+1>& arr) -> std::tuple< std::array<T, Ns+1>...>
	{
		size_t offset = 0;
		size_t offset2 = 0;
		std::tuple<std::array<T, Ns+1>...> res = {
			ToSubStrStorage<Ns,T>(arr.data() + (offset2 = offset,offset += (Ns+1),offset2))...
		};
		return res;
	}

	template<size_t N, typename UIntT>
	constexpr auto ToHexStrStorage(const std::array<UIntT, N>& arr) -> std::array<char, sizeof(UIntT)* 2 * N>
	{
		std::array<char, sizeof(UIntT) * 2 *N> res = {};
		for (size_t i = 0; i < N; ++i)
		{
			for (size_t j = 0; j < sizeof(UIntT); ++j)
			{
				auto v  = static_cast<uint8_t>((arr[i] >> (8 * (sizeof(UIntT) - 1 - j))) & static_cast<UIntT>(0xFF));
				auto v0 = ((v >> 4) & 0x0F);
				auto v1 = ((v >> 0) & 0x0F);
				if (v0 <= 9) {
					res[2 * (sizeof(UIntT) * i + j) + 0] = '0' + v0;
				}
				else if (v0 <= 15) {
					res[2 * (sizeof(UIntT) * i + j) + 0] = 'a' + v0 - 10;
				}
				if (v1 <= 9) {
					res[2 * (sizeof(UIntT) * i + j) + 1] = '0' + v1;
				}
				else if (v1 <= 15) {
					res[2 * (sizeof(UIntT) * i + j) + 1] = 'a' + v1 - 10;
				}
			}
		}
		return res;
	}
	template<typename UIntT, size_t N>
	constexpr auto FromHexStrStorage(const  std::array<char, sizeof(UIntT) * 2 * N>& arr) -> std::array<UIntT, N>
	{
		std::array<UIntT, N> res = {};
		for (size_t i = 0; i < N; ++i)
		{
			res[i] = 0;
			for (size_t j = 0; j < sizeof(UIntT); ++j)
			{
				auto v0 = arr[2 * (sizeof(UIntT) * i + j) + 0];
				auto v1 = arr[2 * (sizeof(UIntT) * i + j) + 1];

				auto u0 = uint8_t(0);
				auto u1 = uint8_t(0);
				if ((v0 >= '0') && (v0 <= '9')) {
					u0 = static_cast<uint8_t>(v0 - '0');
				}
				if ((v0 >= 'A') && (v0 <= 'F')) {
					u0 = static_cast<uint8_t>(static_cast<uint8_t>(v0 - 'A') + 10);
				}
				if ((v0 >= 'a') && (v0 <= 'f')) {
					u0 = static_cast<uint8_t>(static_cast<uint8_t>(v0 - 'a') + 10);
				}

				if ((v1 >= '0') && (v1 <= '9')) {
					u1 = static_cast<uint8_t>(v1 - '0');
				}
				if ((v1 >= 'A') && (v1 <= 'F')) {
					u1 = static_cast<uint8_t>(static_cast<uint8_t>(v1 - 'A') + 10);
				}
				if ((v1 >= 'a') && (v1 <= 'f')) {
					u1 = static_cast<uint8_t>(static_cast<uint8_t>(v1 - 'a') + 10);
				}

				uint8_t uv = (u0 << (uint8_t)4) + u1;
				res[i] += static_cast<UIntT>(uv) << (8 * (sizeof(UIntT) - 1 - j));
			}
		}
		return res;
	}
}
#endif