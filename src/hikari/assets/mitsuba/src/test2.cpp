#include <hikari/assets/mitsuba/scene_importer.h>
#include <hikari/core/camera.h>
#include <hikari/core/film.h>
#include <hikari/core/node.h>
#include <hikari/shape/mesh.h>
#include <hikari/shape/cube.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/rectangle.h>
#include <hikari/camera/perspective.h>
#include <hikari/light/envmap.h>
#include <glm/gtx/string_cast.hpp>
#include <filesystem>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

static inline constexpr char vsSource[] =
R"(#version 330 core
uniform mat4 model = mat4(1.0);
uniform mat4 view  = mat4(1.0);
uniform mat4 proj  = mat4(1.0);
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 uv;
out vec3 frag_normal;
out vec3 frag_uvw;
void main(){
  mat4 model_view = view * model;
  mat3 normal_mat = transpose(inverse(mat3(model_view)));
  gl_Position     = proj * view * model * vec4(position,1.0);
  frag_normal     = normalize(normal_mat * normal);
}
)";

static inline constexpr char fsSource[] =
R"(#version 330 core
in vec3 frag_normal;
in vec3 frag_uvw;
layout(location = 0) out vec3 frag_color;
void main(){
  frag_color = vec3(0.5 *frag_normal + 0.5);
}
)";

struct Uniform {
  hikari::Mat4x4 model;
  hikari::Mat4x4 view;
  hikari::Mat4x4 proj;
};

struct ShapeData {
  void init(const hikari::ShapePtr& shape) {
    if (shape->getID() == hikari::ShapeMesh::ID()) {
      initMesh(std::static_pointer_cast<hikari::ShapeMesh>(shape));
    }else if (shape->getID() == hikari::ShapeRectangle::ID()) {
      initRect(std::static_pointer_cast<hikari::ShapeRectangle>(shape));
    }
    else if (shape->getID() == hikari::ShapeCube::ID()) {
      initCube(std::static_pointer_cast<hikari::ShapeCube>(shape));
    }
    else if (shape->getID() == hikari::ShapeSphere::ID()) {
      initSphere(std::static_pointer_cast<hikari::ShapeSphere>(shape));
    }
  }
  void initMesh(const std::shared_ptr<hikari::ShapeMesh>     & mesh)
  {
    GLuint bffs[4];
    glGenBuffers(4, bffs);
    vbo_position = bffs[0];
    vbo_normal   = bffs[1];
    vbo_uv       = bffs[2];
    ibo          = bffs[3];

    auto pos = mesh->getVertexPositions();
    auto nor = mesh->getVertexNormals();
    auto tex = mesh->getVertexUVs();
    auto idx = mesh->getFaces();


    idx_count = 3*mesh->getFaceCount();

    glGenVertexArrays(1, &vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pos[0]) * pos.size(), pos.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, sizeof(nor[0]) * nor.size(), nor.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tex[0]) * tex.size(), tex.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx[0]) * idx.size(), idx.data(), GL_STATIC_DRAW);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec3), 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec3), 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec2), 0);
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  void initRect(const std::shared_ptr<hikari::ShapeRectangle>& rect){
    GLuint bffs[4];
    glGenBuffers(4, bffs);
    vbo_position = bffs[0];
    vbo_normal = bffs[1];
    vbo_uv = bffs[2];
    ibo = bffs[3];

    auto mesh = rect->createMesh()->convert<hikari::ShapeMesh>();
    auto vertices = mesh->getVertexPositions();
    auto normals = mesh->getVertexNormals();
    auto uvs = mesh->getVertexUVs();
    auto indices = mesh->getFaces();

    idx_count = indices.size();

    glGenVertexArrays(1, &vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, sizeof(normals[0]) * normals.size(), normals.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvs[0]) * uvs.size(), uvs.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices[0]) * indices.size(), indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec3), 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec3), 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec2), 0);
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  void initCube(const std::shared_ptr<hikari::ShapeCube>     & cube) {
    auto mesh = cube->createMesh()->convert<hikari::ShapeMesh>();
    auto vertices = mesh->getVertexPositions();
    auto normals  = mesh->getVertexNormals();
    auto uvs      = mesh->getVertexUVs();
    auto indices  = mesh->getFaces();

    GLuint bffs[4];
    glGenBuffers(4, bffs);
    vbo_position = bffs[0];
    vbo_normal = bffs[1];
    vbo_uv = bffs[2];
    ibo = bffs[3];

    idx_count = indices.size();
    glGenVertexArrays(1, &vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, sizeof(normals[0]) * normals.size(), normals.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvs[0]) * uvs.size(), uvs.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices[0]) * indices.size(), indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec3), 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec3), 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec2), 0);
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  void initSphere(const std::shared_ptr<hikari::ShapeSphere>& sphere) {
    auto mesh = sphere->createMesh()->convert<hikari::ShapeMesh>();
    auto vertices = mesh->getVertexPositions();
    auto normals = mesh->getVertexNormals();
    auto uvs = mesh->getVertexUVs();
    auto indices = mesh->getFaces();

    GLuint bffs[4];
    glGenBuffers(4, bffs);
    vbo_position = bffs[0];
    vbo_normal = bffs[1];
    vbo_uv = bffs[2];
    ibo = bffs[3];

    idx_count = indices.size();
    glGenVertexArrays(1, &vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glBufferData(GL_ARRAY_BUFFER, sizeof(normals[0]) * normals.size(), normals.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvs[0]) * uvs.size(), uvs.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices[0]) * indices.size(), indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec3), 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec3), 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(hikari::Vec2), 0);
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  void free() {
    GLuint bffs[4] = {};
    bffs[0]= vbo_position ;
    bffs[1]= vbo_normal;
    bffs[2]= vbo_uv;
    bffs[3]= ibo;
    glDeleteBuffers(4, bffs);
    glDeleteVertexArrays(1, &vao);
    vbo_position = 0;
    vbo_normal = 0;
    vbo_uv = 0;
    ibo = 0;
    vao = 0;
  }


  void draw() {
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, idx_count, GL_UNSIGNED_INT, nullptr);
  }

  hikari::Mat4x4 model;
  GLuint idx_count;
  GLuint vbo_position;
  GLuint vbo_normal;
  GLuint vbo_uv;
  GLuint ibo;
  GLuint vao;
};

int main() {
  using namespace std::string_literals;
  auto filepath = std::filesystem::path(R"(D:\Users\shums\Documents\C++\Hikari\data\mitsuba\car\scene.xml)");
  auto importer = hikari::MitsubaSceneImporter::create();
  auto scene    = importer->load(filepath.string());
  auto cameras  = scene->getCameras();// カメラ
  auto lights   = scene->getLights();// 光源  
  auto shapes   = scene->getShapes();// 形状
  auto surfaces = importer->getSurfaceMap();// マテリアル
  auto envmap   = hikari::BitmapPtr();
  {
    for (auto& light : lights) {
      auto light_envmap = light->convert<hikari::LightEnvmap>();
      if (light_envmap) {
        envmap = light_envmap->getBitmap();
      }
    }
  }

  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* window = glfwCreateWindow(
    cameras[0]->getFilm()->getWidth(),
    cameras[0]->getFilm()->getHeight(),
    "title",
    nullptr,
    nullptr
  );
  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
    glfwDestroyWindow(window);
    return -1;
  }
  GLuint prog  = []() -> GLuint{
    GLuint prog = 0;
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint valid_programs = true;
    do {
      {
        const char* p = vsSource;
        glShaderSource(vs, 1, &p, nullptr);
        glCompileShader(vs);

        GLint res;
        glGetShaderiv(vs, GL_COMPILE_STATUS, &res);

        char log[1024] = {}; GLsizei size_of_log;
        glGetShaderInfoLog(vs, sizeof(log), &size_of_log, log);

        if (size_of_log > 0) printf("vs log: \"%s\"\n", log);

        if (res != GL_TRUE) { valid_programs = false; break; }
      }
      {
        const char* p = fsSource;
        glShaderSource(fs, 1, &p, nullptr);
        glCompileShader(fs);

        GLint res;
        glGetShaderiv(fs, GL_COMPILE_STATUS, &res);

        char log[1024] = {}; GLsizei size_of_log;
        glGetShaderInfoLog(fs, sizeof(log), &size_of_log, log);

        if (size_of_log > 0) printf("fs log: \"%s\"\n", log);

        if (res != GL_TRUE) { valid_programs = false; break; }
      }
    } while (false);
    if (valid_programs) {
      prog = glCreateProgram();
      glAttachShader(prog, vs);
      glAttachShader(prog, fs);
      glLinkProgram(prog);

      GLint res;
      glGetProgramiv(prog, GL_LINK_STATUS, &res);

      char log[1024] = {}; GLsizei size_of_log;
      glGetProgramInfoLog(prog, sizeof(log), &size_of_log, log);

      if (size_of_log > 0) printf("prog log: \"%s\"\n", log);

      if (res != GL_TRUE) { glDeleteProgram(prog); valid_programs = false; }
    }
    glDeleteShader(vs); glDeleteShader(fs);
    if (!valid_programs) {
      return 0;
    }
    else {
      return prog;
    }
  }();
  
  std::vector<ShapeData> shape_datas = {};
  shape_datas.resize(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto node = shapes[i]->getNode();
    shape_datas[i].model = node->getGlobalTransform().getMat();
    shape_datas[i].init(shapes[i]);
  }

  GLint model_loc = glGetUniformLocation(prog, "model");
  GLint view_loc  = glGetUniformLocation(prog, "view");
  GLint proj_loc  = glGetUniformLocation(prog, "proj");
  // P * V * Mで
  auto sensor_node = cameras[0]->getNode();
  auto perspective = cameras[0]->convert<hikari::CameraPerspective>();
  auto aspect      = ((float)cameras[0]->getFilm()->getWidth()) / ((float)cameras[0]->getFilm()->getHeight());
  // MITSUBAのカメラ座標系からOPENGLのカメラ座標系へ変換
  auto view_matrix = sensor_node->getGlobalTransform().getMat();
  view_matrix[0]  *= -1.0f;
  view_matrix[2]  *= -1.0f;
  view_matrix      = glm::inverse(view_matrix);
  
  auto proj_matrix = hikari::Mat4x4();
  {
    auto op_fov = perspective->getFov();
    auto fovy   = static_cast<float>(0.0f);
    if (op_fov) {
      auto axis = perspective->getFovAxis();
      if (axis == hikari::CameraFovAxis::eSmaller) {
        if (aspect > 1.0f) {// W/H > 1.0f
          axis = hikari::CameraFovAxis::eY;
        }
        else {
          axis = hikari::CameraFovAxis::eX;
        }
      }
      if (axis == hikari::CameraFovAxis::eLarger) {
        if (aspect > 1.0f) {// W/H > 1.0f
          axis = hikari::CameraFovAxis::eX;
        }
        else {
          axis = hikari::CameraFovAxis::eY;
        }
      }

      if (axis == hikari::CameraFovAxis::eX) {
        ///Y |         |X
        ///H |         |
        ///  |_______  |_______Z
        ///     W    X
        auto ax = tanf(0.5f * glm::radians(*op_fov));
        auto ay = ax / aspect;
        fovy = 2.0f * atanf(ay);
      }
      else if (axis == hikari::CameraFovAxis::eY) {
        fovy = glm::radians(*op_fov);
      }
      else {
        throw std::runtime_error("Unsupported Axis Type!");
      }
    }
    else {
      throw std::runtime_error("Unsupported Camera!");
    }

    proj_matrix = glm::perspective(fovy, aspect, perspective->getNearClip(), perspective->getFarClip());
  }


  std::cout << glm::to_string(view_matrix) << std::endl;
  std::cout << glm::to_string(proj_matrix) << std::endl;
  // proj * view * model
  glEnable(GL_DEPTH_TEST);
  // glEnable(GL_CULL_FACE);
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, cameras[0]->getFilm()->getWidth(), cameras[0]->getFilm()->getHeight());
    glClearColor(1.0f, 0.0f,0.0f, 0.0f);

    if (glfwGetKey(window, GLFW_KEY_D)==GLFW_PRESS) {
      view_matrix[3][0] += 0.01f;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
      view_matrix[3][0] -= 0.01f;
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
      view_matrix[3][1] += 0.01f;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
      view_matrix[3][1] -= 0.01f;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
      view_matrix *= glm::rotate(0.01f, glm::vec3(0.0f, 1.0f, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
      view_matrix *= glm::rotate(-0.01f, glm::vec3(0.0f, 1.0f, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
      view_matrix[3][2] += 0.01f;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
      view_matrix[3][2] -= 0.01f;
    }
    printf("%f %f %f\n", view_matrix[3].x, view_matrix[3].y, view_matrix[3].z);

    glUseProgram(prog);
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, (const GLfloat*)&view_matrix);
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, (const GLfloat*)&proj_matrix);
    for (auto& shape_data : shape_datas) {
      glUniformMatrix4fv(model_loc, 1, GL_FALSE, (const GLfloat*)&shape_data.model);
      auto tmp = view_matrix* shape_data.model;
      shape_data.draw();
    }


    glfwSwapBuffers(window);
  }

  for (size_t i = 0; i < shapes.size(); ++i) {
    shape_datas[i].free();
  }

  glDeleteProgram(prog);
  glfwDestroyWindow(window);
  glfwTerminate();
}
