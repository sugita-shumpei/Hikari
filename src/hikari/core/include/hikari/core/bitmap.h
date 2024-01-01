#pragma once
#include <hikari/core/data_type.h>
#include <memory>
namespace hikari {
    enum class BitmapDimension {
      e1D,
      e2D,
      e3D,
      eLayer1D,
      eLayer2D
    };
    enum class BitmapDataType {
      eU8 ,
      eU16,
      eU32,
      eU64,

      eI8 ,
      eI16,
      eI32,
      eI64,

      eF16,
      eF32,
      eF64 
    };
    struct BitmapImageDesc {
      const void* p_data          = nullptr;
      size_t      width_in_bytes  = 1;
      size_t      height          = 1;
      size_t      depth_or_layers = 1;
      size_t      x = 0;
      size_t      y = 0;
      size_t      z = 0;

      auto get(size_t i = 0, size_t j = 0, size_t k = 0)const -> const void* {
        return (const Byte*)p_data+((x + i) + width_in_bytes * (y + j) + width_in_bytes * height * (z + k));
      }
      
    };
    struct Bitmap {
      using DataType  = BitmapDataType;
      using Dimension = BitmapDimension;
      using ImageDesc = BitmapImageDesc;

      static auto create(Dimension dimension, DataType data_type, U32 channel, U32 width, U32 height, U32 depth_or_layers, const BitmapImageDesc* desc = nullptr, bool is_immutable = true) -> std::shared_ptr<Bitmap>;
      static auto create1D(DataType data_type, U32 channel, U32 width, const BitmapImageDesc* desc = nullptr, bool is_immutable = true) -> std::shared_ptr<Bitmap>;
      static auto create2D(DataType data_type, U32 channel, U32 width, U32 height, const BitmapImageDesc* desc = nullptr, bool is_immutable = true) -> std::shared_ptr<Bitmap>;
      static auto create3D(DataType data_type, U32 channel, U32 width, U32 height, U32 depth, const BitmapImageDesc* desc = nullptr, bool is_immutable = true) -> std::shared_ptr<Bitmap>;
      static auto createLayer1D(DataType data_type, U32 channel, U32 width, U32 layers, const BitmapImageDesc* desc = nullptr, bool is_immutable = true) -> std::shared_ptr<Bitmap>;
      static auto createLayer2D(DataType data_type, U32 channel, U32 width, U32 height, U32 layers, const BitmapImageDesc* desc = nullptr, bool is_immutable = true) -> std::shared_ptr<Bitmap>;
      ~Bitmap() noexcept;

      auto getDimension()     const->Dimension;// Dimension
      auto getDataType()      const->DataType;// Dimension
      auto getDataTypeSize()  const->U64;
      auto getChannel()       const->U32;// Dimension
      auto getWidth()         const->U32;// Width 
      auto getHeight()        const->U32;// Height
      auto getDepthOrLayers() const->U32;// Layer

      Bool isImmutable() const;
      // 値を読み取る
      auto getData() const -> const void*;
      Bool getData(U32 x, U32 y, U32 depth_or_layer, void* p_data) const;
      void setData(U32 x, U32 y, U32 depth_or_layer, const void* p_data);
      //
      template<typename T, size_t N>
      Bool getData(U32 x, U32 y, U32 depth_or_layer, std::array<T, N>& data)const {
        if (getChannel() != N) { return false; }
#define HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(TYPE) \
        if constexpr (std::is_same_v<T, TYPE>::value) { if (getDataType() != DataType::e##TYPE) { return false; } }

        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(I8);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(I16);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(I32);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(I64);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(U8);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(U16);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(U32);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(U64);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(F16);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(F32);
        HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH(F64);

#undef HK_CORE_BITMAP_IMPL_GET_DATA_PATTERN_MATCH
        return getData(x, y, depth_or_layer,&data);
      }
      template<typename T, size_t N>
      void setData(U32 x, U32 y, U32 depth_or_layer, const std::array<T, N>& data) {
        if (getChannel() != N) { return; }
#define HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(TYPE) \
        if constexpr (std::is_same_v<T, TYPE>::value) { if (getDataType() != DataType::e##TYPE) { return; } }

        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(I8);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(I16);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(I32);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(I64);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(U8);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(U16);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(U32);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(U64);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(F16);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(F32);
        HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH(F64);

#undef HK_CORE_BITMAP_IMPL_SET_DATA_PATTERN_MATCH
        return setData(x, y, depth_or_layer, &data);
      }
      static auto getDataTypeSize(DataType data_type) -> U64;
    private:
      Bitmap(Dimension dimension, DataType data_type, U32 channel, U32 width, U32 height, U32 depth_or_layers, const ImageDesc* p_desc,bool is_immutable);
    private:
      struct Impl;
      std::unique_ptr<Impl> m_impl;
    };
    using BitmapPtr = std::shared_ptr<Bitmap>;
}
