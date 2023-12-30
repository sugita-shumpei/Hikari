#include <hikari/bsdf/null.h>

auto hikari::BsdfNull::create() -> std::shared_ptr<BsdfNull>
{
    return std::shared_ptr<BsdfNull>(new BsdfNull());
}

hikari::BsdfNull::~BsdfNull()
{
}

hikari::Uuid hikari::BsdfNull::getID() const
{
    return ID();
}

hikari::BsdfNull::BsdfNull():Bsdf()
{
}
