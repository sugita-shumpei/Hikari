#include <gl_viewer.h>
#include <cstdio>
#include <stdexcept>
#include <cstdint>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace hikari {
    namespace test {
        namespace owl {
            namespace testlib {
                struct GLViewerInternalData {
                    hikari::test::owl::testlib::GLViewer* p_viewer;
                    double                                scroll_x;
                    double                                scroll_y;
                };
            }
        }
    }
}

void Internal_Glfw_ScrollCallback(GLFWwindow* window, double x, double y) {
    auto viewer        = (hikari::test::owl::testlib::GLViewer*)glfwGetWindowUserPointer(window);
    auto internal      = (hikari::test::owl::testlib::GLViewerInternalData*)viewer->getInternalData();
    internal->scroll_x = x;
    internal->scroll_y = y;
}

hikari::test::owl::testlib::GLViewer::GLViewer(void* stream_,int width_, int height_) noexcept
    : 
    m_stream{stream_},
    m_width{ width_ },
    m_height{ height_ },
    m_shd{ 0 }, 
    m_tex{ 0 }, 
    m_pbo{ 0 }, 
    m_vao{ 0 }, 
    m_tex_loc{ -1 }, 
    m_update_next_frame{false},
    m_delta_time{0.0f},
    m_internal{ new GLViewerInternalData{ this,0.0f,0.0f } } {
}
hikari::test::owl::testlib::GLViewer::~GLViewer() noexcept {
    GLViewerInternalData* internal_data = (GLViewerInternalData*)m_internal;
    m_internal = nullptr;
    delete m_internal;
}


bool hikari::test::owl::testlib::GLViewer::shouldClose()
{
    GLFWwindow* tmp_window = (GLFWwindow*)m_window;
    return glfwWindowShouldClose(tmp_window);
}

bool hikari::test::owl::testlib::GLViewer::resize(int new_width, int new_height, bool should_clear ) {
    if (m_width != new_width || m_height != new_height || should_clear) {
        bool is_mapped = m_device_ptr != nullptr;
        m_width = new_width; m_height = new_height;
        createFrameResource();
        if (is_mapped) {
            mapFramePtr();
        }
        return true;
    }
    return false;
}
void hikari::test::owl::testlib::GLViewer::render() {
    copyFrameResource();
    drawFrameMesh();
}

void hikari::test::owl::testlib::GLViewer::beginUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    //ImGui::SetNextWindowPos( ImVec2(30, 30)  );
    //ImGui::SetNextWindowSize(ImVec2(200, 250));
}

void hikari::test::owl::testlib::GLViewer::endUI()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void* hikari::test::owl::testlib::GLViewer::mapFramePtr() {
    cudaGraphicsResource* cuda_resource = (cudaGraphicsResource*)m_resource;
    if (!m_resource) { return nullptr; }
    if (m_device_ptr) { return m_device_ptr; }
    cudaGraphicsMapResources(1, &cuda_resource, (CUstream)m_stream);
    size_t size = 0;
    cudaGraphicsResourceGetMappedPointer(&m_device_ptr, &size, cuda_resource);
    return m_device_ptr;
}
void hikari::test::owl::testlib::GLViewer::unmapFramePtr() {
    cudaGraphicsResource* cuda_resource = (cudaGraphicsResource*)m_resource;
    if (!cuda_resource || !m_device_ptr) { return; }
    cudaGraphicsUnmapResources(1, &cuda_resource, (CUstream)m_stream);
    m_device_ptr = nullptr;
}
void* hikari::test::owl::testlib::GLViewer::getFramePtr() { return m_device_ptr; }

void hikari::test::owl::testlib::GLViewer::init()
{
    initGlfw3();
    initGlad();
    initImgui();
    createDummyVAO();
    createShaderProgram();
    createFrameResource();
}

void hikari::test::owl::testlib::GLViewer::free()
{
    deleteFrameResource();
    deleteShaderProgram();
    deleteDummyVAO();
    freeImgui();
    freeGlad();
    freeGlfw3();
}

void hikari::test::owl::testlib::GLViewer::mainLoop()
{
    GLFWwindow* tmp_window = (GLFWwindow*)m_window;
    glfwShowWindow(tmp_window);
    while (!shouldClose()) {
        glfwPollEvents();
        {
            bool is_updated   = m_update_next_frame;
            m_update_next_frame = false;
            int old_width = m_width; int old_height = m_height;
            int tmp_width, tmp_height;
            glfwGetWindowSize(tmp_window, &tmp_width, &tmp_height);
            if (resize(tmp_width, tmp_height)) {
                onResize(old_width,old_height,tmp_width,tmp_height);
                is_updated = true;
            }
            auto io = ImGui::GetIO();
            if (!io.WantCaptureKeyboard) {
                if (glfwGetKey(tmp_window, GLFW_KEY_W) == GLFW_PRESS) {
                    if (onPressKey(KeyType::eW)) { is_updated = true; }
                }
                if (glfwGetKey(tmp_window, GLFW_KEY_A) == GLFW_PRESS) {
                    if (onPressKey(KeyType::eA)) { is_updated = true; }
                }
                if (glfwGetKey(tmp_window, GLFW_KEY_S) == GLFW_PRESS) {
                    if (onPressKey(KeyType::eS)) { is_updated = true; }
                }
                if (glfwGetKey(tmp_window, GLFW_KEY_D) == GLFW_PRESS) {
                    if (onPressKey(KeyType::eD)) { is_updated = true; }
                }
                if (glfwGetKey(tmp_window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                    if (onPressKey(KeyType::eLeft)) { is_updated = true; }
                }
                if (glfwGetKey(tmp_window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                    if (onPressKey(KeyType::eRight)) { is_updated = true; }
                }
                if (glfwGetKey(tmp_window, GLFW_KEY_UP) == GLFW_PRESS) {
                    if (onPressKey(KeyType::eUp)) { is_updated = true; }
                }
                if (glfwGetKey(tmp_window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                    if (onPressKey(KeyType::eDown)) { is_updated = true; }
                }
            }
            if (!io.WantCaptureMouse) {
                if (glfwGetMouseButton(tmp_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                    if (onPressMouseButton(MouseButtonType::eLeft)) { is_updated = true; }
                }
                if (glfwGetMouseButton(tmp_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                    if (onPressMouseButton(MouseButtonType::eRight)) { is_updated = true; }
                }
            }
            if (is_updated) {
                onUpdate();
            }
        }
        {
            void* map_ptr = mapFramePtr();
            onRender(map_ptr);
            unmapFramePtr();
        }
        beginUI();
        render();
        endUI();
        glfwSwapBuffers(tmp_window);
    }
    glfwHideWindow(tmp_window);
}

void hikari::test::owl::testlib::GLViewer::mainLoopWithCallback(
    Pfn_GLViewerResizeCallback           resizeCallback, 
    Pfn_GLViewerPressKeyCallback         pressKeyCallback, 
    Pfn_GLViewerPressMouseButtonCallback pressMouseButtonCallback,
    Pfn_GLViewerMouseScrollCallback      mouseScrollCallback,
    Pfn_GLViewerUpdateCallback           updateCallback,
    Pfn_GLViewerRenderCallback           renderCallback,
    Pfn_GLViewerGuiCallback              guiCallback)
{
    GLFWwindow* tmp_window = (GLFWwindow*)m_window;
    glfwShowWindow(tmp_window);
    bool is_first = true;
    m_delta_time  = -1.0f;
    while (!shouldClose()) {
        glfwPollEvents();
        {
            if (is_first) {
                glfwSetTime(0.0f);
                is_first =  false;
            }
            else {
                m_delta_time = glfwGetTime();
            }

            int old_width = m_width; int old_height = m_height;
            bool is_updated  = m_update_next_frame;
            m_update_next_frame = false;
            int tmp_width, tmp_height;
            glfwGetWindowSize(tmp_window, &tmp_width, &tmp_height);
            if (resize(tmp_width, tmp_height)) {
                if (resizeCallback) resizeCallback(this, old_width, old_height, tmp_width, tmp_height);
                is_updated = true;
            }
            auto io = ImGui::GetIO();
            if (!io.WantCaptureKeyboard){
                if (pressKeyCallback) {
                    if (glfwGetKey(tmp_window, GLFW_KEY_W) == GLFW_PRESS) {
                        if (pressKeyCallback(this, KeyType::eW)) { is_updated = true; }
                    }
                    if (glfwGetKey(tmp_window, GLFW_KEY_A) == GLFW_PRESS) {
                        if (pressKeyCallback(this, KeyType::eA)) { is_updated = true; }
                    }
                    if (glfwGetKey(tmp_window, GLFW_KEY_S) == GLFW_PRESS) {
                        if (pressKeyCallback(this, KeyType::eS)) { is_updated = true; }
                    }
                    if (glfwGetKey(tmp_window, GLFW_KEY_D) == GLFW_PRESS) {
                        if (pressKeyCallback(this, KeyType::eD)) { is_updated = true; }
                    }
                    if (glfwGetKey(tmp_window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                        if (pressKeyCallback(this, KeyType::eLeft)) { is_updated = true; }
                    }
                    if (glfwGetKey(tmp_window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                        if (pressKeyCallback(this, KeyType::eRight)) { is_updated = true; }
                    }
                    if (glfwGetKey(tmp_window, GLFW_KEY_UP) == GLFW_PRESS) {
                        if (pressKeyCallback(this, KeyType::eUp)) { is_updated = true; }
                    }
                    if (glfwGetKey(tmp_window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                        if (pressKeyCallback(this, KeyType::eDown)) { is_updated = true; }
                    }
                }
            }
            if (!io.WantCaptureMouse) {
                if (pressMouseButtonCallback) {
                    if (glfwGetMouseButton(tmp_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                        if (pressMouseButtonCallback(this, MouseButtonType::eLeft)) { is_updated = true; }
                    }
                    if (glfwGetMouseButton(tmp_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                        if (pressMouseButtonCallback(this, MouseButtonType::eRight)) { is_updated = true; }
                    }
                }
            }
            // Mouseの状態更新
            {
                GLViewerInternalData* p_internal = (GLViewerInternalData*)getInternalData();
                if (!io.WantCaptureMouse) {
                    if (p_internal->scroll_x != 0.0f || p_internal->scroll_y != 0.0f) {
                        if (mouseScrollCallback) {
                            if (mouseScrollCallback(this, p_internal->scroll_x, p_internal->scroll_y)) { is_updated = true; }
                        }
                    }
                }
                p_internal->scroll_x = 0.0f;
                p_internal->scroll_y = 0.0f;
            }
            if (is_updated) {
                if (updateCallback) { updateCallback(this); }
            }
        }
        if (renderCallback){
            void* map_ptr = mapFramePtr();
            renderCallback(this, map_ptr);
            unmapFramePtr();
        }
        beginUI();
        if(guiCallback) guiCallback(this);
        render();
        endUI();
        glfwSwapBuffers(tmp_window);
    }
    glfwHideWindow(tmp_window);
}

void hikari::test::owl::testlib::GLViewer::initGlfw3()
{
    glfwInit();
    glfwWindowHint(GLFW_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* tmp_window = glfwCreateWindow(m_width, m_height, "title", nullptr, nullptr);
    glfwMakeContextCurrent(tmp_window);
    glfwSetWindowUserPointer(tmp_window,this);
    glfwSetScrollCallback(tmp_window, Internal_Glfw_ScrollCallback);
    m_window = tmp_window;
}

void hikari::test::owl::testlib::GLViewer::freeGlfw3()
{
    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(static_cast<GLFWwindow*>(m_window));
    glfwTerminate();
}

void hikari::test::owl::testlib::GLViewer::initGlad()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed To Support OpenGL!");
    }
}

void hikari::test::owl::testlib::GLViewer::freeGlad()
{

}

void hikari::test::owl::testlib::GLViewer::initImgui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    ImGui::StyleColorsDark();
    GLFWwindow* tmp_window = (GLFWwindow*)m_window;
    ImGui_ImplGlfw_InitForOpenGL(tmp_window, true);
    ImGui_ImplGlfw_SetCallbacksChainForAllWindows(true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}
void hikari::test::owl::testlib::GLViewer::freeImgui()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
void  hikari::test::owl::testlib::GLViewer::createShaderProgram() {
    // -1    1     3
    const char* vs_source =
        R"(
#version 460 core
vec3 positions[3] = {
vec3(-1.0,-1.0,0.0),
vec3(+3.0,-1.0,0.0),
vec3(-1.0,+3.0,0.0)
};
out vec2 uv;
void main(){
vec3 position = positions[gl_VertexID];
gl_Position   = vec4(position,1.0);
uv            = vec2((position.xy+vec2(1.0))/2.0);
}
)";
    const char* fs_source =
        R"(
#version 460 core
uniform sampler2D tex;
in  vec2 uv;
layout(location=0) out vec4 color;
void main(){
    color = texture(tex,uv);
}
)";

    glDeleteProgram(m_shd);
    auto vs = glCreateShader(GL_VERTEX_SHADER);
    auto fs = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vs, 1, &vs_source, nullptr);
    glShaderSource(fs, 1, &fs_source, nullptr);
    glCompileShader(vs);
    glCompileShader(fs);
    {
        int  log_len = 0;
        char log[1024];
        glGetShaderInfoLog(vs, sizeof log, &log_len, log);
        printf("log vs: %s\n", log);
        glGetShaderInfoLog(fs, sizeof log, &log_len, log);
        printf("log fs: %s\n", log);
    }
    m_shd = glCreateProgram();
    glAttachShader(m_shd, vs);
    glAttachShader(m_shd, fs);
    glLinkProgram(m_shd);
    {
        int  log_len = 0;
        char log[1024];
        glGetProgramInfoLog(m_shd, sizeof log, &log_len, log);
        printf("log pg: %s\n", log);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    m_tex_loc = glGetUniformLocation(m_shd, "tex");
}
void hikari::test::owl::testlib::GLViewer::deleteShaderProgram() {
    glDeleteProgram(m_shd);
    m_shd = 0;
    m_tex_loc = 0;
}
void hikari::test::owl::testlib::GLViewer::createDummyVAO() {
    glDeleteVertexArrays(1, &m_vao);
    glGenVertexArrays(1, &m_vao);
}
void hikari::test::owl::testlib::GLViewer::deleteDummyVAO() {
    glDeleteVertexArrays(1, &m_vao);
    m_vao = 0;
}
void hikari::test::owl::testlib::GLViewer::drawFrameMesh() {
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.0, 0.0, 1.0, 1.0);
    glViewport(0, 0, m_width, m_height);
    glUseProgram(m_shd);
    glBindVertexArray(m_vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glUniform1i(m_tex_loc, 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
void hikari::test::owl::testlib::GLViewer::createFrameResource() {
    cudaGraphicsResource* cuda_resource = (cudaGraphicsResource*)m_resource;
    if (m_resource) {
        if (m_device_ptr) {
            cudaGraphicsUnmapResources(1, &cuda_resource, (CUstream)m_stream);
            m_device_ptr = nullptr;
        }
        cudaGraphicsUnregisterResource(cuda_resource);
        m_resource = nullptr;
    }

    glDeleteBuffers(1, &m_pbo);
    glDeleteTextures(1, &m_tex);

    glGenBuffers(1, &m_pbo);
    glGenTextures(1, &m_tex);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(uint32_t), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


    cudaGraphicsGLRegisterBuffer(&cuda_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard);
    m_resource = cuda_resource;
}
void hikari::test::owl::testlib::GLViewer::deleteFrameResource() {
    if (m_resource) {
        cudaGraphicsResource* cuda_resource = (cudaGraphicsResource*)m_resource;
        if (m_device_ptr) {
            cudaGraphicsUnmapResources(1, &cuda_resource, (CUstream)m_stream);
            m_device_ptr = nullptr;
        }
        cudaGraphicsUnregisterResource(cuda_resource);
        m_resource = nullptr;
    }
    glDeleteBuffers(1, &m_pbo);
    glDeleteTextures(1, &m_tex);
    m_pbo = 0; m_tex = 0;
}
void hikari::test::owl::testlib::GLViewer::copyFrameResource() {
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void hikari::test::owl::testlib::GLViewer::run()
{
    init();
    mainLoop();
    free();
}

void hikari::test::owl::testlib::GLViewer::runWithCallback(
    void*                                pUserData,
    Pfn_GLViewerResizeCallback           resizeCallback, 
    Pfn_GLViewerPressKeyCallback         pressKeyCallback, 
    Pfn_GLViewerPressMouseButtonCallback pressMouseButtonCallback,
    Pfn_GLViewerMouseScrollCallback      mouseScrollCallback,
    Pfn_GLViewerUpdateCallback           updateCallback, 
    Pfn_GLViewerRenderCallback           renderCallback,
    Pfn_GLViewerGuiCallback              guiCallback)
{
    void* tmp_userptr = m_user_ptr;
    if (pUserData) { m_user_ptr = pUserData; }
    init();
    mainLoopWithCallback(resizeCallback, pressKeyCallback, pressMouseButtonCallback, mouseScrollCallback, updateCallback, renderCallback,guiCallback);
    free();
    m_user_ptr = tmp_userptr;
}

void hikari::test::owl::testlib::GLViewer::setUserPtr(void* ptr)
{
    m_user_ptr = ptr;
}

void* hikari::test::owl::testlib::GLViewer::getUserPtr()
{
    return m_user_ptr;
}

void* hikari::test::owl::testlib::GLViewer::getInternalData()
{
    return m_internal;
}

void hikari::test::owl::testlib::GLViewer::updateNextFrame()
{
    m_update_next_frame = true;
}

void hikari::test::owl::testlib::GLViewer::getCursorPosition(double& cursor_pos_x, double& cursor_pos_y) const
{
    GLFWwindow* tmp_window = (GLFWwindow*)m_window;
    glfwGetCursorPos(tmp_window, &cursor_pos_x, &cursor_pos_y);
}

void hikari::test::owl::testlib::GLViewer::getWindowSize(int& width_, int& height_)const
{
    width_  = m_width;
    height_ = m_height;
}
