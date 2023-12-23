#pragma once
#include <hikari/core/data_type.h>
#include <memory>
namespace hikari {
  enum class CameraFovAxis {
    eX,       // X軸
    eY,       // Y軸
    eDiagonal,// 対角線(未実装)
    eSmaller, // フィルム幅が小さいほうの軸
    eLarger   // フィルム幅が大きいほうの軸
  };
  struct Node;
  struct Film;
  struct Medium;
  struct Camera : public std::enable_shared_from_this<Camera> {
    virtual ~Camera() noexcept;
    auto getNode()   -> std::shared_ptr<Node>;
    auto getFilm() -> std::shared_ptr<Film>;
    void setFilm(const std::shared_ptr<Film>& film);
    auto getMedium() -> std::shared_ptr<Medium>;
    void setMedium(const std::shared_ptr<Medium>& medium);
    virtual Uuid getID() const = 0;
    template<typename DeriveType>
    auto convert() -> std::shared_ptr<DeriveType> {
      if (DeriveType::ID() == getID()) {
        return std::static_pointer_cast<DeriveType>(shared_from_this());
      }
      else {
        return nullptr;
      }
    }
  protected:
    Camera();
  private:
    friend struct Node;
    void onAttach(const std::shared_ptr<Node>& node);
    void onDetach();
  private:
    std::weak_ptr<Node> m_node;
    std::shared_ptr<Film> m_film;
    std::shared_ptr<Medium> m_medium;
  };
  using  CameraPtr = std::shared_ptr<Camera>;
}
