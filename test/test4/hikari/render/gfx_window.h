#pragma once
#include <array>
#include <hikari/core/common.h>
#include <hikari/render/gfx_common.h>
  
namespace hikari {
  inline namespace render {
    struct GFXWindowObject;
    struct GFXWindow;
    struct GFXWindowObject {
      virtual ~GFXWindowObject() noexcept {}
      virtual auto getAPI () const -> GFXAPI = 0;
      virtual auto getHandle() const -> void* = 0;
      virtual bool isResizable() const = 0;
      virtual bool isClosed() const = 0;
      virtual bool isFocused() const = 0;
      virtual bool isVisible() const = 0;
      virtual bool isIconified() const = 0;
      virtual bool isBorderless() const = 0;
      virtual void setResizable(bool resizable) = 0;
      virtual void setVisible(bool visible) = 0;
      virtual void setIconified(bool iconified) = 0;
      virtual void setBorderless(bool borderless) = 0;
      virtual auto getTitle() const->std::string = 0;
      virtual void setTitle(const std::string& s) = 0;
      virtual auto getClipboard() const->std::string = 0;
      virtual void setClipboard(const std::string& s) = 0;
      virtual auto getSize() const->std::array<uint32_t, 2> = 0;
      virtual auto getPosition() const->std::array<uint32_t, 2> = 0;
      virtual auto getFramebufferSize() const->std::array<uint32_t, 2> = 0;
      virtual void setSize(const std::array<uint32_t, 2>& size)  = 0;
      virtual void setPosition(const std::array<uint32_t, 2>& pos) = 0;
      inline void show() { setVisible(true); }
      inline void hide() { setVisible(false); }
    };
#define HK_RENDER_GFX_WINDOW_METHOD_OVERRIDES(BASE) \
      using impl_type = BASE; \
      using type = typename impl_type::type; \
      using impl_type::operator!; \
      using impl_type::operator bool; \
      using impl_type::getAPI; \
      using impl_type::getHandle; \
      using impl_type::getObject; \
      using impl_type::isClosed; \
      using impl_type::isFocused; \
      using impl_type::isVisible; \
      using impl_type::isIconified; \
      using impl_type::isBorderless; \
      using impl_type::getClipboard; \
      using impl_type::getTitle; \
      using impl_type::setVisible; \
      using impl_type::setIconified; \
      using impl_type::setClipboard; \
      using impl_type::setBorderless; \
      using impl_type::setTitle; \
      using impl_type::getSize; \
      using impl_type::getFramebufferSize; \
      using impl_type::setSize; \
      using impl_type::getPosition; \
      using impl_type::setPosition; \
      using impl_type::show; \
      using impl_type::hide
      
    template<typename GFXWindowObjectT, std::enable_if_t<std::is_base_of_v<GFXWindowObject, GFXWindowObjectT>, nullptr_t> = nullptr>
    struct GFXWindowImpl : protected GFXWrapper<GFXWindowObjectT> {
      using impl_type = GFXWrapper<GFXWindowObjectT>;
      using type = typename impl_type::type;
      GFXWindowImpl() noexcept :impl_type() {}
      GFXWindowImpl(nullptr_t) noexcept :impl_type(nullptr) {}
      GFXWindowImpl(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
      template<typename GFXWindowLike, std::enable_if_t<std::is_base_of_v<GFXWindowObjectT, typename GFXWindowLike::type>, nullptr_t > = nullptr>
      GFXWindowImpl(const GFXWindowLike& v) noexcept :impl_type(v.getObject()) {}
#define HK_GFX_WINDOW_TYPE_ARRAY_U32_2 std::array<uint32_t, 2>
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getAPI, GFXAPI, GFXAPI::eUnknown);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF_NO_CAST(getHandle, void*, nullptr);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(isResizable, bool, false);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(isClosed , bool, true);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(isVisible, bool, false);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(isFocused, bool, false);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(isIconified, bool, false);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(isBorderless, bool, true);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getClipboard, std::string, "");
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getTitle, std::string, "");
      HK_METHOD_OVERLOAD_SETTER_LIKE(setResizable, bool);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setVisible,bool);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setIconified, bool);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setBorderless, bool);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setClipboard, std::string);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setTitle, std::string);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getSize, HK_GFX_WINDOW_TYPE_ARRAY_U32_2, {});
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getFramebufferSize, HK_GFX_WINDOW_TYPE_ARRAY_U32_2, {});
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getPosition, HK_GFX_WINDOW_TYPE_ARRAY_U32_2, {});
      HK_METHOD_OVERLOAD_SETTER_LIKE(setSize, HK_GFX_WINDOW_TYPE_ARRAY_U32_2);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setPosition, HK_GFX_WINDOW_TYPE_ARRAY_U32_2);
      HK_METHOD_OVERLOAD_STATE_SHIFT_LIKE(show);
      HK_METHOD_OVERLOAD_STATE_SHIFT_LIKE(hide);
#undef  HK_GFX_WINDOW_TYPE_ARRAY_U32_2
      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::getObject;
    protected:
      using impl_type::setObject;
    };
    struct GFXWindow : protected GFXWindowImpl<GFXWindowObject> {
      HK_RENDER_GFX_WINDOW_METHOD_OVERRIDES(GFXWindowImpl<GFXWindowObject>);
      GFXWindow() noexcept :impl_type() {}
      GFXWindow(const GFXWindow& v) noexcept : impl_type(v.getObject()) {}
      GFXWindow(nullptr_t) noexcept :impl_type(nullptr) {}
      GFXWindow(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
      template<typename GFXWindowLike, std::enable_if_t<std::is_base_of_v<type, typename GFXWindowLike::type>, nullptr_t > = nullptr>
      GFXWindow(const GFXWindowLike& v) noexcept :impl_type(v.getObject()) {}
      GFXWindow& operator=(const GFXWindow& v) noexcept { if (this != &v) { setObject(v.getObject()); }return*this; }
      GFXWindow& operator=(const std::shared_ptr<type>& obj) noexcept { setObject(obj); return*this; }
      GFXWindow& operator=(nullptr_t) noexcept { setObject(nullptr); return*this; }
      template<typename GFXWindowLike, std::enable_if_t<std::is_base_of_v<type, typename GFXWindowLike::type>, nullptr_t > = nullptr>
      GFXWindow& operator=(const GFXWindowLike& v) noexcept { setObject(v.getObject()); return*this; }
    };
    struct GFXWindowManager {
      static auto getInstance() -> GFXWindowManager&;
      ~GFXWindowManager()noexcept;
      //void addWindow(const GFXWindow& window);
      //void popWindow(const GFXWindow& window);
      void update();
    private:
      GFXWindowManager() noexcept;
      GFXWindowManager(const GFXWindowManager&) = delete;
      GFXWindowManager(GFXWindowManager&&) = delete;
      GFXWindowManager& operator=(const GFXWindowManager&) = delete;
      GFXWindowManager& operator=(GFXWindowManager&&) = delete;
    private:
      //std::unordered_map<GFXWindowObject*, std::shared_ptr<GFXWindowObject>> m_windows;
    };
  }
}
