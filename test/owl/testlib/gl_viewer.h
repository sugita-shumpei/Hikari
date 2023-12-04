#pragma once
#include <owl/owl.h>
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
                using  Pfn_GLViewerUpdateCallback           = void(*)(GLViewer*);
                using  Pfn_GLViewerRenderCallback           = void(*)(GLViewer*,void*);
                using  Pfn_GLViewerGuiCallback              = void(*)(GLViewer*);

                struct GLViewer {
                    GLViewer(OWLContext context_, int width_, int height_) noexcept;
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
                        Pfn_GLViewerUpdateCallback           updateCallback,
                        Pfn_GLViewerRenderCallback           renderCallback,
                        Pfn_GLViewerGuiCallback              guiCallback
                    );
                    void    getCursorPosition(double& cursor_pos_x, double& cursor_pos_y)const;
                    void    getWindowSize(int& width, int& height)const;
                    void*   getUserPtr();
                    void    updateNextFrame();
                protected:
                    void    setUserPtr(void* ptr);
                    virtual bool onResize(int old_width, int old_height,int new_width, int new_height) { return false; }
                    virtual bool onPressKey(KeyType key)                          { return false; }
                    virtual bool onPressMouseButton(MouseButtonType mouse_button) { return false; }
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
                    OWLContext context    = nullptr;
                    void*      window     = nullptr;
                    void*      resource   = nullptr;
                    void*      device_ptr = nullptr;
                    void*      user_ptr   = nullptr;
                    int32_t    width ;
                    int32_t    height;
                    uint32_t   vao;
                    uint32_t   shd;
                    uint32_t   tex;
                    uint32_t   pbo;
                    int32_t    tex_loc;
                    bool       update_next_frame;
                };
            }
        }
    }
}