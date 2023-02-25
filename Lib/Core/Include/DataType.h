#ifndef HIKARI_H_DATATYPE_H
#define HIKARI_H_DATATYPE_H
#include <cstdint>
namespace Hikari
{
	using TChar   = char;
	using TUChar  = unsigned char;
	using TSChar  = signed char;

	static_assert(sizeof(TChar ) == 1);
	static_assert(sizeof(TUChar) == 1);
	static_assert(sizeof(TSChar) == 1);

	using TFloat32 = float ;
	using TFloat64 = double;

	static_assert(sizeof(TFloat32) == 4);
	static_assert(sizeof(TFloat64) == 8);

	using TSInt8  = std::int8_t ;
	using TSInt16 = std::int16_t;
	using TSInt32 = std::int32_t;
	using TSInt64 = std::int64_t;

	static_assert(sizeof(TSInt8)  == 1);
	static_assert(sizeof(TSInt16) == 2);
	static_assert(sizeof(TSInt32) == 4);
	static_assert(sizeof(TSInt64) == 8);

	using TUInt8  = std::uint8_t ;
	using TUInt16 = std::uint16_t;
	using TUInt32 = std::uint32_t;
	using TUInt64 = std::uint64_t;

	static_assert(sizeof(TUInt8)  == 1);
	static_assert(sizeof(TUInt16) == 2);
	static_assert(sizeof(TUInt32) == 4);
	static_assert(sizeof(TUInt64) == 8);

	using TSizeT  = size_t;

	using TBoolT  = bool;

	static_assert(sizeof(TBoolT) == 1);

	using TUIntPtr = std::uintptr_t;
	using TSIntPtr = std::intptr_t;
}
#endif
