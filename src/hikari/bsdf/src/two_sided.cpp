#include <hikari/bsdf/two_sided.h>

auto hikari::BsdfTwoSided::create() -> std::shared_ptr<BsdfTwoSided>
{
    return std::shared_ptr<BsdfTwoSided>(new BsdfTwoSided());
}

hikari::BsdfTwoSided::~BsdfTwoSided()
{
}

auto hikari::BsdfTwoSided::getBsdf() -> BsdfPtr
{
    return m_bsdfs[0];
}

void hikari::BsdfTwoSided::setBsdf(const BsdfPtr& bsdf)
{
  m_bsdfs[0] = bsdf;
  m_bsdfs[1] = nullptr;
}

auto hikari::BsdfTwoSided::getBsdfs()->std::array<BsdfPtr, 2>
{
    return m_bsdfs;
}

void hikari::BsdfTwoSided::setBsdfs(const std::array<BsdfPtr, 2>& bsdfs)
{
  if (bsdfs[0] == nullptr || bsdfs[1] == nullptr) { return; }
  if (bsdfs[0] == bsdfs[1]) {
    m_bsdfs[0] = bsdfs[0];
    m_bsdfs[1] = nullptr;
  }
  else {
    m_bsdfs = bsdfs;
  }
}

bool hikari::BsdfTwoSided::isSeparate() const
{
  if (!m_bsdfs[0]) { return false; }
  return m_bsdfs[1]!=nullptr;
}

hikari::Uuid hikari::BsdfTwoSided::getID() const
{
    return ID();
}

hikari::BsdfTwoSided::BsdfTwoSided()
  :Bsdf(),m_bsdfs{nullptr,nullptr}
{
}
