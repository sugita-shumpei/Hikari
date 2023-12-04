#include <gl_viewer.h>
#include <cstdio>
#include <stdexcept>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

hikari::test::owl::testlib::GLViewer::GLViewer(OWLContext context_, int width_, int height_) noexcept
    : 
    context{ context_ },
    width{ width_ },
    height{ height_ },
    shd{ 0 }, tex{ 0 }, pbo{ 0 }, vao{ 0 }, tex_loc{ -1 }, update_next_frame{false} {
}
hikari::test::owl::testlib::GLViewer::~GLViewer() noexcept {
}

bool hikari::test::owl::testlib::GLViewer::shouldClose()
{
    GLFWwindow* tmp_window = (GLFWwindow*)window;
    return glfwWindowShouldClose(tmp_window);
}

bool hikari::test::owl::testlib::GLViewer::resize(int new_width, int new_height, bool should_clear ) {
    if (width != new_width || height != new_height || should_clear) {
        bool is_mapped = device_ptr != nullptr;
        width = new_width; height = new_height;
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
    cudaGraphicsResource* cuda_resource = (cudaGraphicsResource*)resource;
    if (!resource) { return nullptr; }
    if (device_ptr) { return device_ptr; }
    cudaGraphicsMapResources(1, &cuda_resource, owlContextGetStream(context, 0));
    size_t size = 0;
    cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, cuda_resource);
    return device_ptr;
}
void hikari::test::owl::testlib::GLViewer::unmapFramePtr() {
    cudaGraphicsResource* cuda_resource = (cudaGraphicsResource*)resource;
    if (!cuda_resource || !device_ptr) { return; }
    cudaGraphicsUnmapResources(1, &cuda_resource, owlContextGetStream(context, 0));
    device_ptr = nullptr;
}
void* hikari::test::owl::testlib::GLViewer::getFramePtr() { return device_ptr; }

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
    GLFWwindow* tmp_window = (GLFWwindow*)window;
    glfwShowWindow(tmp_window);
    while (!shouldClose()) {
        glfwPollEvents();
        {
            bool is_updated   = update_next_frame;
            update_next_frame = false;
            int old_width = width; int old_height = height;
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
    Pfn_GLViewerUpdateCallback           updateCallback,
    Pfn_GLViewerRenderCallback           renderCallback,
    Pfn_GLViewerGuiCallback              guiCallback)
{
    GLFWwindow* tmp_window = (GLFWwindow*)window;
    glfwShowWindow(tmp_window);
    while (!shouldClose()) {
        glfwPollEvents();
        {
            int old_width = width; int old_height = height;
            bool is_updated  = update_next_frame;
            update_next_frame = false;
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
    GLFWwindow* tmp_window = glfwCreateWindow(width, height, "title", nullptr, nullptr);
    glfwMakeContextCurrent(tmp_window);
    window = tmp_window;
}

void hikari::test::owl::testlib::GLViewer::freeGlfw3()
{
    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(static_cast<GLFWwindow*>(window));
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
    GLFWwindow* tmp_window = (GLFWwindow*)window;
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

    glDeleteProgram(shd);
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
    shd = glCreateProgram();
    glAttachShader(shd, vs);
    glAttachShader(shd, fs);
    glLinkProgram(shd);
    {
        int  log_len = 0;
        char log[1024];
        glGetProgramInfoLog(shd, sizeof log, &log_len, log);
        printf("log pg: %s\n", log);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    tex_loc = glGetUniformLocation(shd, "tex");
}
void hikari::test::owl::testlib::GLViewer::deleteShaderProgram() {
    glDeleteProgram(shd);
    shd = 0;
    tex_loc = 0;
}
void hikari::test::owl::testlib::GLViewer::createDummyVAO() {
    glDeleteVertexArrays(1, &vao);
    glGenVertexArrays(1, &vao);
}
void hikari::test::owl::testlib::GLViewer::deleteDummyVAO() {
    glDeleteVertexArrays(1, &vao);
    vao = 0;
}
void hikari::test::owl::testlib::GLViewer::drawFrameMesh() {
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.0, 0.0, 1.0, 1.0);
    glViewport(0, 0, width, height);
    glUseProgram(shd);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(tex_loc, 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
void hikari::test::owl::testlib::GLViewer::createFrameResource() {
    cudaGraphicsResource* cuda_resource = (cudaGraphicsResource*)resource;
    if (resource) {
        if (device_ptr) {
            cudaGraphicsUnmapResources(1, &cuda_resource, owlContextGetStream(context, 0));
            device_ptr = nullptr;
        }
        cudaGraphicsUnregisterResource(cuda_resource);
        resource = nullptr;
    }

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);

    glGenBuffers(1, &pbo);
    glGenTextures(1, &tex);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uint32_t), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


    cudaGraphicsGLRegisterBuffer(&cuda_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    resource = cuda_resource;
}
void hikari::test::owl::testlib::GLViewer::deleteFrameResource() {
    if (resource) {
        cudaGraphicsResource* cuda_resource = (cudaGraphicsResource*)resource;
        if (device_ptr) {
            cudaGraphicsUnmapResources(1, &cuda_resource, owlContextGetStream(context, 0));
            device_ptr = nullptr;
        }
        cudaGraphicsUnregisterResource(cuda_resource);
        resource = nullptr;
    }
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    pbo = 0; tex = 0;
}
void hikari::test::owl::testlib::GLViewer::copyFrameResource() {
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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
    Pfn_GLViewerUpdateCallback           updateCallback, 
    Pfn_GLViewerRenderCallback           renderCallback,
    Pfn_GLViewerGuiCallback              guiCallback)
{
    void* tmp_userptr = user_ptr;
    if (pUserData) { user_ptr = pUserData; }
    init();
    mainLoopWithCallback(resizeCallback, pressKeyCallback, pressMouseButtonCallback, updateCallback, renderCallback,guiCallback);
    free();
    user_ptr = tmp_userptr;
}

void hikari::test::owl::testlib::GLViewer::setUserPtr(void* ptr)
{
    user_ptr = ptr;
}

void* hikari::test::owl::testlib::GLViewer::getUserPtr()
{
    return user_ptr;
}

void hikari::test::owl::testlib::GLViewer::updateNextFrame()
{
    update_next_frame = true;
}

void hikari::test::owl::testlib::GLViewer::getCursorPosition(double& cursor_pos_x, double& cursor_pos_y) const
{
    GLFWwindow* tmp_window = (GLFWwindow*)window;
    glfwGetCursorPos(tmp_window, &cursor_pos_x, &cursor_pos_y);
}

void hikari::test::owl::testlib::GLViewer::getWindowSize(int& width_, int& height_)const
{
    width_ = width;
    height_ = height;
}
