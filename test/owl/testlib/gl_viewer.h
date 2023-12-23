#pragma once
#include <cstdint>
#include <string>
#include <imgui.h>
namespace hikari {
    namespace test {
        namespace owl {
            namespace testlib {
                enum class KeyType {
                    eW,
                    eA,
                    eS,
                    eD,
                    eUp,
                    eDown ,
                    eLeft ,
                    eRight,
                    eCount
                };
                enum class MouseButtonType {
                    eLeft ,
                    eRight,
                    eCount
                };
                struct GLViewer;
                using  Pfn_GLViewerResizeCallback           = bool(*)(GLViewer*, int old_width, int old_height, int new_width, int new_height);
                using  Pfn_GLViewerPressKeyCallback         = bool(*)(GLViewer*, KeyType         key);
                using  Pfn_GLViewerPressMouseButtonCallback = bool(*)(GLViewer*, MouseButtonType mouse_button);
                using  Pfn_GLViewerMouseScrollCallback      = bool(*)(GLViewer*, double x, double y);
                using  Pfn_GLViewerUpdateCallback           = void(*)(GLViewer*);
                using  Pfn_GLViewerRenderCallback           = void(*)(GLViewer*,void*);
                using  Pfn_GLViewerGuiCallback              = void(*)(GLViewer*);

                struct GLViewer {
                    GLViewer(void* m_stream, int width_, int height_) noexcept;
                    virtual ~GLViewer() noexcept ;

                    GLViewer(const GLViewer& ) noexcept = delete;
                    GLViewer& operator=(const GLViewer&) noexcept = delete;

                    GLViewer(GLViewer&&) noexcept = delete;
                    GLViewer& operator=(GLViewer&&) noexcept = delete;

                    void    run();
                    void    runWithCallback(
                        void*                                pUserData,
                        Pfn_GLViewerResizeCallback           resizeCallback,
                        Pfn_GLViewerPressKeyCallback         pressKeyCallback,
                        Pfn_GLViewerPressMouseButtonCallback pressMouseButtonCallback,
                        Pfn_GLViewerMouseScrollCallback      mouseScrollCallback,
                        Pfn_GLViewerUpdateCallback           updateCallback,
                        Pfn_GLViewerRenderCallback           renderCallback,
                        Pfn_GLViewerGuiCallback              guiCallback
                    );

                    void   getCursorPosition(double& cursor_pos_x, double& cursor_pos_y)const;
                    void   getWindowSize(int& width, int& m_height)const;
                    void*  getUserPtr();
                    void*  getInternalData();
                    void   updateNextFrame();
                    double getTime()const { return m_delta_time; }
                    
                protected:
                    void         setUserPtr(void* ptr);
                    virtual bool onResize(int old_width, int old_height,int new_width, int new_height) { return false; }
                    virtual bool onPressKey(KeyType key)                            { return false; }
                    virtual bool onPressMouseButton(MouseButtonType mouse_button)   { return false; }
                    virtual bool onMouseScroll(double delta_x, double      delta_y) { return false; }
                    virtual void onUpdate()                {}
                    virtual void onRender(void* frame_ptr) {}
                private:
                    void  init();
                    void  free();
                    void  mainLoop();
                    void  mainLoopWithCallback(
                        Pfn_GLViewerResizeCallback           resizeCallback,
                        Pfn_GLViewerPressKeyCallback         pressKeyCallback,
                        Pfn_GLViewerPressMouseButtonCallback pressMouseButtonCallback,
                        Pfn_GLViewerMouseScrollCallback      mouseScrollCallback,
                        Pfn_GLViewerUpdateCallback           updateCallback,
                        Pfn_GLViewerRenderCallback           renderCallback,
                        Pfn_GLViewerGuiCallback              guiCallback
                    );
                    void  initGlfw3();
                    void  freeGlfw3();
                    void  initGlad() ;
                    void  freeGlad() ;
                    void  initImgui();
                    void  freeImgui();
                    void  createShaderProgram();
                    void  deleteShaderProgram() ;
                    void  createDummyVAO();
                    void  deleteDummyVAO();
                    void  drawFrameMesh();
                    void  createFrameResource();
                    void  deleteFrameResource();
                    void  copyFrameResource() ;
                    bool  resize(int new_width, int new_height, bool should_clear = false);
                    void  render();
                    void  beginUI();
                    void  endUI();
                    bool  shouldClose();
                    void* mapFramePtr();
                    void  unmapFramePtr();
                    void* getFramePtr();
                private:
                    void*    m_stream     = nullptr;
                    void*    m_window     = nullptr;
                    void*    m_resource   = nullptr;
                    void*    m_device_ptr = nullptr;
                    void*    m_user_ptr   = nullptr;
                    void*    m_internal   = nullptr;
                    int32_t  m_width ;
                    int32_t  m_height;
                    uint32_t m_vao;
                    uint32_t m_shd;
                    uint32_t m_tex;
                    uint32_t m_pbo;
                    int32_t  m_tex_loc;
                    double   m_delta_time;
                    bool     m_update_next_frame;
                };
            }
        }
    }
}
