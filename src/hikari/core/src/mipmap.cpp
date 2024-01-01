#include <hikari/core/mipmap.h>

auto hikari::Mipmap::create(Dimension dimension, DataType data_type, U32 channel, U32 levels, U32 width, U32 height, U32 depth_or_layers, const std::vector<ImageDesc>& descs, bool is_immutable) -> std::shared_ptr<Mipmap>
{
  std::vector<std::array<U32, 3>> read_descs = { {width,height,depth_or_layers} };
  if (levels > 1) {
    for (size_t i = 1; i < levels; ++i) {
      read_descs[i][0] = read_descs[i - 1][0] / 2;
      if ((dimension == hikari::MipmapDimension::e2D) || (dimension == hikari::MipmapDimension::eLayer2D)) {
        read_descs[i][1] = read_descs[i - 1][1] / 2;
      }
      else {
        read_descs[i][1] = read_descs[i - 1][1];
      }
      if (dimension == hikari::MipmapDimension::e3D) {
        read_descs[i][2] = read_descs[i - 1][2] / 2;
      }
      else {
        read_descs[i][2] = read_descs[i - 1][2];
      }
      if (read_descs[i][0] == 1) { break; }
      if ((dimension == hikari::MipmapDimension::e2D) || (dimension == hikari::MipmapDimension::eLayer2D)) { if (read_descs[i][1] == 1) { break; } }
      if (dimension == hikari::MipmapDimension::e3D) { if (read_descs[i][2] == 1) { break; } }
    }
  }
  levels = read_descs.size();
  // 全体のうちread descに対応するモノをreadする
  auto res= std::shared_ptr<Mipmap>(new Mipmap());
  // BITMAPをアロケートする
  res->m_bitmaps.reserve(levels);
  // ____w____w/2_
  // |       |   |
  // h       |___|
  // |       |_|
  // |_______|
  //
  size_t levels_with_data = std::min<size_t>(descs.size(), levels);
  for (size_t i = 0; i < levels_with_data; ++i) {
    auto tmp = Bitmap::create(dimension, data_type, channel, read_descs[i][0], read_descs[i][1], read_descs[i][2], &descs[i], is_immutable);
    if (!tmp) { return nullptr; }
    res->m_bitmaps.push_back(tmp);
  }
  for (size_t i = levels_with_data; i < levels; ++i) {
    auto tmp = Bitmap::create(dimension, data_type, channel, read_descs[i][0], read_descs[i][1], read_descs[i][2], nullptr  , is_immutable);
    if (!tmp) { return nullptr; }
    res->m_bitmaps.push_back(tmp);
  }
  return res;
}

auto hikari::Mipmap::create1D(DataType data_type, U32 channel, U32 levels, U32 width, const std::vector<ImageDesc>& descs, bool is_immutable) -> std::shared_ptr<Mipmap>
{
  return create(Dimension::e1D, data_type, channel, levels, width, 1, 1, descs, is_immutable);
}

auto hikari::Mipmap::create2D(DataType data_type, U32 channel, U32 levels, U32 width, U32 height, const std::vector<ImageDesc>& descs, bool is_immutable) -> std::shared_ptr<Mipmap>
{
  return create(Dimension::e2D, data_type, channel, levels, width, height, 1, descs, is_immutable);
}

auto hikari::Mipmap::create3D(DataType data_type, U32 channel, U32 levels, U32 width, U32 height, U32 depth, const std::vector<ImageDesc>& descs, bool is_immutable) -> std::shared_ptr<Mipmap>
{
  return create(Dimension::e3D, data_type, channel, levels, width, height, depth, descs, is_immutable);
}

auto hikari::Mipmap::createLayer1D(DataType data_type, U32 channel, U32 levels, U32 width, U32 layers, const std::vector<ImageDesc>& descs, bool is_immutable) -> std::shared_ptr<Mipmap>
{
  return create(Dimension::eLayer1D, data_type, channel, levels, width, 1, layers, descs, is_immutable);
}

auto hikari::Mipmap::createLayer2D(DataType data_type, U32 channel, U32 levels, U32 width, U32 height, U32 layers, const std::vector<ImageDesc>& descs, bool is_immutable) -> std::shared_ptr<Mipmap>
{
    return create(Dimension::eLayer2D,data_type,channel,levels,width,height,layers, descs,is_immutable);
}

hikari::Mipmap::~Mipmap()noexcept
{
}

auto hikari::Mipmap::getWidth() const -> U32
{
  return m_bitmaps[0]->getWidth();
}

auto hikari::Mipmap::getHeight() const -> U32
{
    return m_bitmaps[0]->getHeight();
}

auto hikari::Mipmap::getDepthOrLayers() const -> U32
{
    return m_bitmaps[0]->getDepthOrLayers();
}

auto hikari::Mipmap::getLevels() const -> U32
{
    return m_bitmaps.size();
}

auto hikari::Mipmap::getDataType() const -> DataType
{
  return m_bitmaps[0]->getDataType();
}

auto hikari::Mipmap::getDimension() const -> Dimension
{
  return m_bitmaps[0]->getDimension();
}

auto hikari::Mipmap::getChannel() const -> U32
{
  return m_bitmaps[0]->getChannel();
}

auto hikari::Mipmap::getImage(U32 idx) -> BitmapPtr
{
    if (idx >= m_bitmaps.size()) { return nullptr;}
    return m_bitmaps[idx];
}

hikari::Bool hikari::Mipmap::isImmutable() const
{
    return m_bitmaps[0]->isImmutable();
}

hikari::Bool hikari::Mipmap::getData(U32 level, U32 x, U32 y, U32 depth_or_layer, void* p_data) const
{
    if (level >= m_bitmaps.size()) { return false; }
    return m_bitmaps[level]->getData(x,y,depth_or_layer,p_data);
}

void hikari::Mipmap::setData(U32 level, U32 x, U32 y, U32 depth_or_layer, const void* p_data)
{
  if (level >= m_bitmaps.size()) { return; }
  return m_bitmaps[level]->setData(x, y, depth_or_layer, p_data);
}

hikari::Mipmap::Mipmap()
  :m_bitmaps{}
{
}
