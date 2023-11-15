#include <gl_viewer.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cstdio>

hikari::test::owl::testlib::GLViewer::GLViewer(OWLContext context_, int width_, int height_) noexcept
    : 
    context{ context_ },
    width{ width_ },
    height{ height_ },
    shd{ 0 }, tex{ 0 }, pbo{ 0 } {
    createDummyVAO();
    createShaderProgram();
    createFrameResource();
}
hikari::test::owl::testlib::GLViewer::~GLViewer() noexcept {
    deleteFrameResource();
    deleteShaderProgram();
    deleteDummyVAO();
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

bool hikari::test::owl::testlib::loadGLLoader(GLloadproc load_proc)
{
    return gladLoadGLLoader(load_proc);
}
