#include <hikari/core/bitmap.h>
#include <cstdlib>

struct hikari::Bitmap::Impl {
  Impl(Dimension dimension, DataType data_type, U32 channel, U32 width, U32 height, U32 depth_or_layers, const ImageDesc* p_desc, Bool is_immutable)
    :m_dimension{ dimension }, m_data_type{ data_type }, m_channel{ channel }, m_width{ width }, m_height{ height }, m_depth_or_layers{ depth_or_layers }, m_is_immutable{ is_immutable }
  {
#define HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(TYPE) \
  case DataType::e##TYPE: \
    { \
      m_data = std::unique_ptr<Byte[],AlignedAllocator>((Byte*)AlignedAllocator::alloc<TYPE>(width*height*channel*depth_or_layers),AlignedAllocator()); \
      break; \
    }
    switch (m_data_type)
    {
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(U8);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(U16);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(U32);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(U64);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(I8);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(I16);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(I32);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(I64);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(F16);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(F32);
      HK_BITMAP_IMPL_CONSTRUCTOR_PATTERN_MATCH(F64);
    default:
      break;
    }

    if (p_desc) {
      auto data_type_size = Bitmap::getDataTypeSize(data_type);
      for (size_t k = 0; k < depth_or_layers; ++k) {
        for (size_t j = 0; j < height; ++j) {
          std::memcpy(m_data.get() + (width * height * k + width * j) * data_type_size * channel, p_desc->get(0,j,k), width * data_type_size * channel);
        }
      }
    }
  }
  ~Impl() {

  }

  struct AlignedAllocator {
    template<typename T>
    static Byte* alloc(size_t count) {
#ifdef _MSC_VER
      return (Byte*)_aligned_malloc(count * sizeof(T), alignof(T));
#else
      return (Byte*)std::aligned_alloc(count * sizeof(T), alignof(T));
#endif
    }
    static void release(Byte* ptr) {
#ifdef _MSC_VER
      _aligned_free(ptr);
#else
      delete ptr;
#endif
    }

    void operator()(Byte* ptr) const {
      release(ptr);
    }
  };

  Dimension               m_dimension;
  DataType                m_data_type;
  U32                     m_channel;
  U32                     m_width;
  U32                     m_height;
  U32                     m_depth_or_layers;
  Bool                    m_is_immutable;
  std::unique_ptr<Byte[], AlignedAllocator> m_data;
};

auto hikari::Bitmap::create(Dimension dimension, DataType data_type, U32 channel, U32 width, U32 height, U32 depth_or_layers, const ImageDesc* p_desc, Bool is_immutable) -> std::shared_ptr<Bitmap>
{
  if (p_desc) {
    if (!p_desc->p_data) { return nullptr; }
    if (p_desc->x + width *channel* getDataTypeSize(data_type) > p_desc->width_in_bytes ) { return nullptr; }
    if (p_desc->y + height                                     > p_desc->height         ) { return nullptr; }
    if (p_desc->z + depth_or_layers                            > p_desc->depth_or_layers) { return nullptr; }
  }
  auto res = std::shared_ptr<Bitmap>(new Bitmap(dimension, data_type, channel, width, height, depth_or_layers, p_desc, is_immutable));
  return res;
}

auto hikari::Bitmap::create1D(DataType data_type, U32 channel, U32 width, const ImageDesc* p_desc, Bool is_immutable) -> std::shared_ptr<Bitmap>
{
  return create(Dimension::e1D, data_type, channel, width, 1, 1, p_desc,is_immutable);
}

auto hikari::Bitmap::create2D(DataType data_type, U32 channel, U32 width, U32 height, const ImageDesc* p_desc, Bool is_immutable) -> std::shared_ptr<Bitmap>
{
  return create(Dimension::e2D, data_type, channel, width, height, 1, p_desc, is_immutable);
}

auto hikari::Bitmap::create3D(DataType data_type, U32 channel, U32 width, U32 height, U32 depth, const ImageDesc* p_desc, Bool is_immutable) -> std::shared_ptr<Bitmap>
{
  return create(Dimension::e3D, data_type, channel, width, height, depth, p_desc, is_immutable);
}

auto hikari::Bitmap::createLayer1D(DataType data_type, U32 channel, U32 width, U32 layers, const ImageDesc* p_desc, Bool is_immutable) -> std::shared_ptr<Bitmap>
{
  return create(Dimension::eLayer1D, data_type, channel, width, 1, layers, p_desc, is_immutable);
}

auto hikari::Bitmap::createLayer2D(DataType data_type, U32 channel, U32 width, U32 height, U32 layers, const ImageDesc* p_desc, Bool is_immutable) -> std::shared_ptr<Bitmap>
{
  return create(Dimension::eLayer2D, data_type, channel, width, height, layers, p_desc, is_immutable);
}

hikari::Bitmap::~Bitmap() noexcept
{
}

auto hikari::Bitmap::getWidth() const -> U32
{
  return m_impl->m_width;
}

auto hikari::Bitmap::getHeight() const -> U32
{
  return m_impl->m_height;
}

auto hikari::Bitmap::getDepthOrLayers() const -> U32
{
  return m_impl->m_depth_or_layers;
}

hikari::Bool hikari::Bitmap::isImmutable() const
{
  return m_impl->m_is_immutable;
}

auto hikari::Bitmap::getData() const -> const void*
{
    return m_impl->m_data.get();
}

hikari::Bool hikari::Bitmap::getData(U32 x, U32 y, U32 depth_or_layer, void* p_data) const
{
  if (x >= m_impl->m_width) { return false; }
  if (y >= m_impl->m_height) { return false; }
  if (depth_or_layer >= m_impl->m_depth_or_layers) { return false; }
  auto data_type_size = getDataTypeSize();
  const auto data = m_impl->m_data.get() + data_type_size * m_impl->m_channel * (m_impl->m_width * m_impl->m_height * depth_or_layer + y * m_impl->m_width + x);
  std::memcpy(p_data, data, data_type_size);
  return true;
}

void hikari::Bitmap::setData(U32 x, U32 y, U32 depth_or_layer, const void* p_data)
{
  if (m_impl->m_is_immutable) { return; }
  if (x >= m_impl->m_width) { return; }
  if (y >= m_impl->m_height) { return; }
  if (depth_or_layer >= m_impl->m_depth_or_layers) { return; }
  auto data_type_size = getDataTypeSize();
  const auto data = m_impl->m_data.get() + data_type_size * m_impl->m_channel * (m_impl->m_width * m_impl->m_height * depth_or_layer + y * m_impl->m_width + x);
  std::memcpy(data, p_data, data_type_size);
}

hikari::Bitmap::Bitmap(Dimension dimension, DataType data_type, U32 channel, U32 width, U32 height, U32 depth_or_layers, const ImageDesc* p_desc, bool is_immutable)
  :m_impl{new Impl(dimension,data_type,channel,width,height,depth_or_layers,p_desc,is_immutable)}{


}

auto hikari::Bitmap::getDataTypeSize(DataType data_type) -> U64
{
  switch (data_type)
  {

#define HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(TYPE) \
        case DataType::e##TYPE: return sizeof(TYPE);

    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(I8);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(I16);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(I32);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(I64);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(U8);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(U16);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(U32);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(U64);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(F16);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(F32);
    HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH(F64);

#undef HK_CORE_BITMAP_IMPL_GET_DATA_SIZE_PATTERN_MATCH
  default: return 0;
  };
}

auto hikari::Bitmap::getDimension() const -> Dimension
{
  return m_impl->m_dimension;
}

auto hikari::Bitmap::getChannel() const -> U32
{
  return m_impl->m_channel;
}

auto hikari::Bitmap::getDataType() const -> DataType
{
  return m_impl->m_data_type;
}

auto hikari::Bitmap::getDataTypeSize() const -> U64
{
  return getDataTypeSize(m_impl->m_data_type);
}
