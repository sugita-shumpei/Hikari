#pragma once
#include <owl/owl.h>
namespace hikari {
    namespace test {
        namespace owl {
            namespace testlib {
                struct GLViewer {
                    GLViewer(OWLContext context_, int width_, int height_) noexcept;
                    ~GLViewer() noexcept ;

                    GLViewer(const GLViewer& ) noexcept = delete;
                    GLViewer& operator=(const GLViewer&) noexcept = delete;

                    GLViewer(GLViewer&&) noexcept = delete;
                    GLViewer& operator=(GLViewer&&) noexcept = delete;
                    // 画面をリサイズする
                    bool  resize(int new_width, int new_height, bool should_clear = false) ;
                    void  render() ;

                    void* mapFramePtr();
                    void  unmapFramePtr();
                    void* getFramePtr() ;
                private:
                    void createShaderProgram();
                    void deleteShaderProgram() ;
                    void createDummyVAO();
                    void deleteDummyVAO();
                    void drawFrameMesh();
                    void createFrameResource();
                    void deleteFrameResource() ;
                    void copyFrameResource() ;
                private:
                    OWLContext            context    = nullptr;
                    void*                 resource   = nullptr;
                    void*                 device_ptr = nullptr;
                    int32_t               width ;
                    int32_t               height;
                    uint32_t              vao;
                    uint32_t              shd;
                    uint32_t              tex;
                    uint32_t              pbo;
                    int32_t               tex_loc;
                };
                typedef void* (*GLloadproc)(const char* name);
                bool loadGLLoader(GLloadproc load_proc);
            }
        }
    }
}