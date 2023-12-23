#pragma once
#include <hikari/core/data_type.h>
#include <memory>
namespace hikari {
  struct Node;
  struct Shape : public std::enable_shared_from_this<Shape> {
    virtual ~Shape() noexcept;
    void    setFlipNormals(Bool flip_normals);
    auto    getFlipNormals() const -> Bool;
    auto    getNode() -> std::shared_ptr<Node>;
    virtual Uuid getID() const = 0;
    template<typename DeriveType>
    auto    convert() -> std::shared_ptr<DeriveType>{
      if (DeriveType::ID() == getID()) {
        return std::static_pointer_cast<DeriveType>(shared_from_this());
      }
      else {
        return nullptr;
      }
    }
  protected:
    Shape();
  private:
    friend struct Node;
    void onAttach(const std::shared_ptr<Node>& node);
    void onDetach();
  private:
    std::weak_ptr<Node> m_node;
    Bool m_flip_normals = false;
  };
  using ShapePtr = std::shared_ptr<Shape>;
}
