#pragma once
#include <shared_mutex>
#include <mutex>
#include <array>
#include <optional>
#include <hikari/core/data_types.h>
#include <hikari/window/common.h>
#include <hikari/graphics/system.h>
#include <hikari/graphics/common.h>
#include <hikari/graphics/opengl/instance.h>
namespace hikari {
  struct Window {
  private:
    static inline constexpr size_t k_cur_index = 0;
    static inline constexpr size_t k_new_index = 0;
  public:
    void setSize(const std::array<U32, 2>& size) { std::lock_guard lk(m_mtx_update); m_cmd_size = size; }
    auto getSize() const->std::array<U32, 2> {  return m_size[k_cur_index]; }
    auto getFramebufferSize() const->std::array<U32, 2> { return m_frame_buffer_size[k_cur_index]; }
    auto getPosition() const->std::array<I32, 2> {  return m_position[k_cur_index]; }
    void setPosition(const std::array<I32, 2>& position) { std::lock_guard lk(m_mtx_update); m_cmd_position = position; }
    auto getKey(KeyInput input) -> KeyStateFlags { return m_key_states[k_cur_index][(U32)input]; }
    auto getKey(KeyInput input, KeyModFlags& mods) -> KeyStateFlags { mods = m_key_mods[k_cur_index]; return m_key_states[k_cur_index][(U32)input]; }
    void setTitle(const std::string& title) { std::lock_guard lk(m_mtx_update); m_cmd_title = title;  }
    auto getTitle() const -> std::string { return m_title; }
    void setFloating(Bool is_floating) { std::lock_guard lk(m_mtx_update); m_cmd_floating = is_floating;}
    void setVisible(Bool is_visible) { std::lock_guard lk(m_mtx_update); m_cmd_visible = is_visible; }
    void setResizable(Bool is_resizable) { std::lock_guard lk(m_mtx_update); m_cmd_resizable = is_resizable; }
    void setFullscreen(Bool is_fullscreen) { std::lock_guard lk(m_mtx_update); m_cmd_fullscreen = is_fullscreen; }
    Bool isClose() const { return m_is_close; }
    Bool isFloating() const { return m_is_floating; }
    Bool isVisible() const { return m_is_visible; }
    Bool isResizable() const { return m_is_resizable; }
    Bool isFullscreen() const { return m_is_fullscreen; }
    void update() {
      std::lock_guard lk(m_mtx_update);
      if (m_cmd_title) {
        glfwSetWindowTitle(m_window, m_cmd_title->c_str());
        m_title = *m_cmd_title;
      }
      if (m_cmd_visible) {
        if (*m_cmd_visible) { glfwShowWindow(m_window); } else { glfwHideWindow(m_window); }
        m_is_visible = *m_cmd_visible;
      }
      if (m_cmd_resizable) {
        glfwSetWindowAttrib(m_window, GLFW_RESIZABLE, *m_cmd_resizable);
        m_is_resizable = *m_cmd_resizable;
      }
      if (m_cmd_size) {
        glfwSetWindowSize(m_window, (*m_cmd_size)[0], (*m_cmd_size)[1]);
      }
      if (m_cmd_position) {
        glfwSetWindowSize(m_window, (*m_cmd_position)[0], (*m_cmd_position)[1]);
      }
      if (m_cmd_floating) {
        glfwSetWindowAttrib(m_window, GLFW_FLOATING, *m_cmd_floating);
        m_is_floating = *m_cmd_floating;
      }
      if (m_cmd_fullscreen) {
        bool new_fullscreen = *m_cmd_fullscreen;
        if (new_fullscreen && !m_is_fullscreen) {
          auto monitor      = glfwGetPrimaryMonitor();
          auto videomode    = glfwGetVideoMode(monitor);
          glfwSetWindowMonitor(m_window, monitor, 0, 0, videomode->width, videomode->height, videomode->refreshRate);
          m_is_fullscreen   = true;
        }
        else if (!new_fullscreen && m_is_fullscreen) {
          auto monitor      = glfwGetWindowMonitor(m_window);
          auto videomode    = glfwGetVideoMode(monitor);
          glfwSetWindowMonitor(m_window, nullptr, videomode->width/4, videomode->height/ 4, videomode->width/2, videomode->height/2, 0);
          m_is_fullscreen   = false;
        }
      }
      m_is_close                       = glfwWindowShouldClose(m_window);
      m_size[k_cur_index]              = m_size[k_new_index];
      m_frame_buffer_size[k_cur_index] = m_frame_buffer_size[k_new_index];
      m_position[k_cur_index]          = m_position[k_new_index]  ;
      m_key_states[k_cur_index]        = m_key_states[k_new_index];
      m_key_states[k_new_index]        = {};
      m_key_mods[k_cur_index]          = m_key_mods[k_new_index]  ;
      m_cmd_size       = std::nullopt;
      m_cmd_position   = std::nullopt;
      m_cmd_title      = std::nullopt;
      m_cmd_floating   = std::nullopt;
      m_cmd_visible    = std::nullopt;
      m_cmd_resizable  = std::nullopt;
      m_cmd_fullscreen = std::nullopt;
    }
    auto getHandle() const -> GLFWwindow* { return m_window; }
  private:
     friend class WindowSystem;
     Window(
       const std::string& title,
       U32 width, U32    height,
       I32 pos_x, I32     pos_y,
       GraphicsAPIType api_type,
       Bool is_floating   ,
       Bool is_resizable ,
       Bool is_visible    ,
       Bool is_fullscreen 
     ) noexcept{
       GLFWwindow* share = nullptr;
       if (api_type == GraphicsAPIType::eOpenGL) {
         auto& graphics = hikari::GraphicsSystem::getInstance();
         graphics.initGraphics(GraphicsAPIType::eOpenGL);
         auto opengl = graphics.getGraphics<hikari::GraphicsOpenGLInstance>();
         glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
         glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
         glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
         glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
         glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
         share = opengl->getHandle();
       }
       else {
         glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
       }
       {
         glfwWindowHint(GLFW_FLOATING , is_floating);
         glfwWindowHint(GLFW_VISIBLE  , is_visible);
         glfwWindowHint(GLFW_RESIZABLE, is_resizable);
         if (is_fullscreen) {
           auto monitor    = glfwGetPrimaryMonitor();
           auto video_mode = glfwGetVideoMode(monitor);
           glfwWindowHint(GLFW_RED_BITS    , video_mode->redBits);
           glfwWindowHint(GLFW_GREEN_BITS  , video_mode->greenBits);
           glfwWindowHint(GLFW_BLUE_BITS   , video_mode->blueBits);
           glfwWindowHint(GLFW_REFRESH_RATE, video_mode->refreshRate);
           m_window   = glfwCreateWindow(video_mode->width, video_mode->height, title.c_str(), monitor, share);
           m_size     = { std::array<U32, 2>{ (U32)video_mode->width, (U32)video_mode->height },std::array<U32, 2>{ (U32)video_mode->width, (U32)video_mode->height} };
           m_position = { std::array<I32,2>{ 0,0 }, std::array<I32,2>{ 0,0 } };
         }
         else {
           m_window   = glfwCreateWindow(width, height, title.c_str(), nullptr, share);
           glfwSetWindowPos(m_window, pos_x, pos_y);
           m_size     = { std::array<U32, 2>{ width, height },std::array<U32, 2>{ width, height} };
           m_position = { std::array<I32,2>{ pos_x,pos_y },std::array<I32,2>{ pos_x,pos_y } };
         }
         glfwDefaultWindowHints();
         I32 w; I32 h;
         glfwGetFramebufferSize(m_window, &w, &h);
         m_frame_buffer_size = { std::array<U32, 2>{(U32)w,(U32)h }, std::array<U32, 2>{(U32)w,(U32)h } };
         m_title = title;
       }
       m_title             = title;
       m_cmd_size          = std::nullopt;
       m_cmd_position      = std::nullopt;
       m_cmd_title         = std::nullopt;
       m_cmd_floating      = std::nullopt;
       m_cmd_visible       = std::nullopt;
       m_cmd_resizable     = std::nullopt;
       m_cmd_fullscreen    = std::nullopt;
       m_key_states        = {};
       m_key_mods          = {};
       m_is_close          = false;
       m_is_floating       = is_floating;
       m_is_visible        = is_visible;
       m_is_resizable      = is_resizable;
       m_is_fullscreen     = is_fullscreen;
       glfwSetWindowUserPointer      (m_window, this);
       glfwSetWindowSizeCallback     (m_window, size_callback);
       glfwSetFramebufferSizeCallback(m_window, fb_size_callback);
       glfwSetWindowPosCallback      (m_window, position_callback);
       glfwSetKeyCallback            (m_window, key_callback);
     }
     ~Window() noexcept {
       if (m_window) {
         glfwDestroyWindow(m_window);
         m_window = nullptr;
       }
    }
  private:
    static void size_callback(GLFWwindow* window, int w, int h){
      if (!window) { return; }
      auto handle = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
      if (handle) {
        auto& size = handle->getNewFramebufferSize();
        if (size[0] != w || size[1] != h) {
          std::cerr << "window size: [" << w << ", " << h << "]" << std::endl;
        }
        size[0] = w;
        size[1] = h;
      }
    }
    static void fb_size_callback(GLFWwindow* window, int w, int h) {
      if (!window) { return; }
      auto handle = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
      if (handle) {
        auto& fb_size = handle->getNewFramebufferSize();
        if (fb_size[0] != w || fb_size[1] != h) {
          std::cerr << " frame size: [" << w << ", " << h << "]" << std::endl;
        }
        fb_size[0] = w;
        fb_size[1] = h;
      }
    }
    static void position_callback(GLFWwindow* window, int x, int y) {
      if (!window) { return; }
      auto handle = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
      if (handle) {
        auto& position  = handle->getNewPosition();
        if (position[0] != x || position[1] != y) {
          std::cerr << "window move: [" << x << ", " << y << "]" << std::endl;
        }
        position[0] = x;
        position[1] = y;
      }
    }
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
      if (!window) { return; }
      auto handle = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
      if (handle) {
        auto key_input = convertInt2KeyInput(key);
        auto key_mods  = convertInt2KeyMods(mods);
        handle->getNewKeyMods() = key_mods;
        if (action == GLFW_PRESS  ) {
          handle->getNewKey(key_input) = (KeyStateFlagBits::ePress| KeyStateFlagBits::eUpdate);
          return;
        }
        if (action == GLFW_RELEASE) {
          handle->getNewKey(key_input) = (KeyStateFlagBits::eRelease   | KeyStateFlagBits::eUpdate);
          return;
        }
        if (action == GLFW_REPEAT ) {
          handle->getNewKey(key_input) = (handle->getCurKey(key_input) & ~KeyStateFlagBits::eUpdate);
          return;
        }
      }
    }
  private:
    inline auto getNewSize() noexcept -> std::array<U32, 2>& { return m_size[k_new_index]; }
    inline auto getNewFramebufferSize() noexcept -> std::array<U32, 2>& { return m_frame_buffer_size[k_new_index]; }
    inline auto getNewPosition() noexcept -> std::array<I32, 2>& { return m_position[k_new_index]; }
    inline auto getNewKeys() noexcept -> std::array<KeyStateFlags, (U32)KeyInput::eCount>& { return m_key_states[k_new_index]; }
    inline auto getCurKeys() noexcept -> std::array<KeyStateFlags, (U32)KeyInput::eCount>& { return m_key_states[k_cur_index]; }
    inline auto getNewKey(KeyInput i) noexcept -> KeyStateFlags& { return getNewKeys()[(U32)i]; }
    inline auto getCurKey(KeyInput i) noexcept -> KeyStateFlags& { return getCurKeys()[(U32)i]; }
    inline auto getNewKeyMods() noexcept -> KeyModFlags&{ return m_key_mods[k_new_index]; }
  private:
    // Thread SafeなAPI設計をしたい
    GLFWwindow* m_window;
    mutable std::recursive_mutex m_mtx_update;
    std::optional<std::array<U32, 2>> m_cmd_size;
    std::optional<std::array<I32, 2>> m_cmd_position;
    std::optional<std::string> m_cmd_title;
    std::optional<bool> m_cmd_floating;
    std::optional<bool> m_cmd_resizable;
    std::optional<bool> m_cmd_visible;
    std::optional<bool> m_cmd_fullscreen;
    std::array<std::array<U32, 2>,2> m_size;
    std::array<std::array<U32, 2>,2> m_frame_buffer_size;
    std::array<std::array<I32, 2>,2> m_position;
    std::array<std::array<KeyStateFlags, (U32)KeyInput::eCount>,2> m_key_states;
    std::array<KeyModFlags, 2> m_key_mods;
    std::string m_title;
    Bool m_is_close;
    Bool m_is_floating ;
    Bool m_is_resizable ;
    Bool m_is_visible ;
    Bool m_is_fullscreen;
  };
}
