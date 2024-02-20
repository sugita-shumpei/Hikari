#include <hikari/framework/core/blob.h>


auto hikari::core::DataBlob::getBufferPointer() const -> const Byte*
{
  auto obj = getObject();
  return obj ? obj->getBufferPointer() : nullptr;
}

auto hikari::core::DataBlob::getBufferSize() const -> const U64
{
  auto obj = getObject();
  return obj ? obj->getBufferSize():0;
}

auto hikari::core::FileBlob::getBufferPointer() const -> const Byte*
{
  auto obj = getObject();
  return obj ? obj->getBufferPointer() : nullptr;
}

auto hikari::core::FileBlob::getBufferSize() const -> const U64
{
  auto obj = getObject();
  return obj ? obj->getBufferSize() : 0;
}

auto hikari::core::FileBlob::getFilePath() const -> Str
{
  auto obj = getObject();
  return obj ? obj->getFilePath() : "";
}
