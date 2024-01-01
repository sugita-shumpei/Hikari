#pragma once
#include <hikari/core/bitmap.h>
#include <hikari/core/data_type.h>
#include <memory>
#include <vector>
namespace hikari {
    using MipmapDimension = BitmapDimension;
    using MipmapDataType  = BitmapDataType;
    using MipmapImageDesc = BitmapImageDesc;
    struct Mipmap {
      using Dimension = MipmapDimension;
      using DataType  = MipmapDataType;
      using ImageDesc = MipmapImageDesc;
      static auto create(Dimension dimension, DataType data_type, U32 channel, U32 levels, U32 width, U32 height, U32 depth_or_layers, const std::vector<ImageDesc>& descs = {}, bool is_immutable = true) -> std::shared_ptr<Mipmap>;
      static auto create1D(DataType data_type, U32 channel, U32 levels, U32 width, const std::vector<ImageDesc>& descs = {}, bool is_immutable = true) -> std::shared_ptr<Mipmap>;
      static auto create2D(DataType data_type, U32 channel, U32 levels, U32 width, U32 height, const std::vector<ImageDesc>& descs = {}, bool is_immutable = true) -> std::shared_ptr<Mipmap>;
      static auto create3D(DataType data_type, U32 channel, U32 levels, U32 width, U32 height, U32 depth, const std::vector<ImageDesc>& descs = {}, bool is_immutable = true) -> std::shared_ptr<Mipmap>;
      static auto createLayer1D(DataType data_type, U32 channel, U32 levels, U32 width, U32 layers, const std::vector<ImageDesc>& descs = {}, bool is_immutable = true) -> std::shared_ptr<Mipmap>;
      static auto createLayer2D(DataType data_type, U32 channel, U32 levels, U32 width, U32 height, U32 layers, const std::vector<ImageDesc>& descs = {}, bool is_immutable = true) -> std::shared_ptr<Mipmap>;
      virtual ~Mipmap()noexcept;

      auto getWidth()         const->U32;// Width 
      auto getHeight()        const->U32;// Height
      auto getDepthOrLayers() const->U32;// Layer
      auto getLevels()        const->U32;// Levels
      auto getDataType()      const->DataType;
      auto getDimension()     const->Dimension;
      auto getChannel()       const->U32;
      auto getImage(U32 idx)->BitmapPtr ;// Bitmap

      Bool isImmutable() const;
      // 値を読み取る
      Bool getData(U32 level, U32 x, U32 y, U32 depth_or_layer, void* p_data) const;
      void setData(U32 level, U32 x, U32 y, U32 depth_or_layer, const void* p_data);

      //
      template<typename T, size_t N>
      Bool getData(U32 level, U32 x, U32 y, U32 depth_or_layer, std::array<T, N>& data)const {
        if (getChannel() != N) { return false; }
#define HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(TYPE) \
        if constexpr (std::is_same_v<T, TYPE>::value) { if (getDataType() != DataType::e##TYPE) { return false; } }

        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(I8);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(I16);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(I32);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(I64);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(U8);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(U16);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(U32);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(U64);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(F16);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(F32);
        HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH(F64);

#undef HK_CORE_MIPMAP_IMPL_GET_DATA_PATTERN_MATCH
        return getData(level,x, y, depth_or_layer, &data);
      }
      template<typename T, size_t N>
      void setData(U32 level, U32 x, U32 y, U32 depth_or_layer, const std::array<T, N>& data) {
        if (getChannel() != N) { return; }
#define HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(TYPE) \
        if constexpr (std::is_same_v<T, TYPE>::value) { if (getDataType()  != DataType::e##TYPE) { return; } }

        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(I8);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(I16);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(I32);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(I64);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(U8);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(U16);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(U32);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(U64);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(F16);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(F32);
        HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH(F64);

#undef HK_CORE_MIPMAP_IMPL_SET_DATA_PATTERN_MATCH
        return setData(level, x, y, depth_or_layer, &data);
      }
    private:
      Mipmap();
      std::vector<BitmapPtr> m_bitmaps;
    };
    using  MipmapPtr = std::shared_ptr<Mipmap>;
}
