#pragma once

#include "Shader.h"
#include "Camera.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Singleton.h"
#include "Texture.h"
#include <unordered_map>
#include "ByteBufferBuilder.h"


namespace Render
{
enum ValueType{
    Float,
    Int,
    Bool,
    Vec2,
    Vec2v,
    Vec3,
    Vec3v,
    Vec4,
    Vec4v,
    Mat2,
    Mat3,
    Mat4
};

enum ObjectType
{
    Triangle = GL_TRIANGLES,
    Point = GL_POINTS,
    Line = GL_LINES
};


class Render : public Singleton<Render>
{

protected:
public:
    ~Render() = default;
    Render() = default;

    typedef std::shared_ptr<Render> ptr;
    void init(unsigned int scr_width = 800, unsigned int scr_height = 600);
    std::vector<unsigned int> addPlanetObject(const uint8_t* particles, const int structSize,
        const int particleNum, unsigned int* planet_id,
        const char* vertexPath, const char* fragmentPath,
        const char* geometryPath = nullptr);

    void addStarObject(const std::vector<float>& vertexs, const std::vector<unsigned int>& incides,
        unsigned int *star_id,
        const char* vertexPath, const char* fragmentPath,
        const char* geometryPath = nullptr);
    void addCubeObject(const char* vertexPath, char* fragmentPath,
        const char* texture_path, const char* texture_post_fix);
    void addStarTexture(const char* path);
    void addCubeTexture(const char* path, const char* postFix);

    void update();
    void render(unsigned int current_vbo);
    void SaveFrameBufferToImage(int frame_id, std::string path = "./images/");
    void cleanup();
    void CreateCamera(glm::vec3 position, glm::vec3 up = { 0.0f, 1.0f, 0.0f }, glm::vec3 front = {0.0f, 0.0f, -1.0f});
    void SetShaderValue(const std::string& name, const void* value, ValueType type);
    bool shouldClose();
    static std::shared_ptr<Render> GetInstance();

    GLFWwindow* window;
    Camera::ptr m_camera;

    std::unordered_map<ObjectType, std::vector<Shader::ptr>> m_shaders;
    std::unordered_map<ObjectType, std::vector<unsigned int>> m_VBOs;
    std::vector<unsigned int> m_planet_VAOs;
    std::unordered_map<unsigned int, unsigned int> m_planet_vertexs_counts;
    ByteBufferBuilder m_planet_vertices;

    std::unordered_map<unsigned int, unsigned int> m_star_vertexs_counts;
    std::vector<unsigned int> m_star_VAOs;
    std::vector<unsigned int> m_star_EBOs;

    unsigned int m_cube_VAO;
    unsigned int m_cube_VBO;

    Texture2D::ptr m_star_texture;
    Shader::ptr m_cube_shader;
    TextureCube::ptr m_cube_texture;
    bool m_use_texture;
    bool m_use_cube_texture;

    unsigned int scr_width;
    unsigned int scr_height;
    static float lastX;
    static float lastY;
    static bool firstMouse;
    bool mousePressed = false;

    float deltaTime = 0.01f;
    float lastFrame = 0.0f;

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    void processInput(GLFWwindow* window);

};

}
