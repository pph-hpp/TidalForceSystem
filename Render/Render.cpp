
#include "Render.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace Render
{
    float Render::lastX;
    float Render::lastY;

    Render::ptr Render::GetInstance(){
        static std::shared_ptr<Render> _instance = std::make_shared<Render>();
        return _instance;
    }
    

    void Render::init(unsigned int SCR_WIDTH, unsigned int SCR_HEIGHT)
    {
        this->scr_width = SCR_WIDTH;
        this->scr_height = SCR_HEIGHT;
        this->lastX = (float)scr_width / 2.0f;
        this->lastY = (float)scr_height / 2.0f;
        this->m_shaders.clear();

        m_VBOs.clear();
        m_planet_VAOs.clear();
        m_planet_vertexs_counts.clear();
        m_star_vertexs_counts.clear();
        m_star_VAOs.clear();
        m_star_EBOs.clear();
        this->m_use_texture = false;
        this->m_use_cube_texture = false;

        m_star_texture = nullptr;
        m_cube_shader = nullptr;
        m_cube_texture = nullptr;

        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        this->window = glfwCreateWindow(scr_width, scr_height, "TidalForce", NULL, NULL);
        if (window == NULL)
        {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetScrollCallback(window, scroll_callback);

        //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        if (glewInit() != GLEW_OK)
        {
            std::cout << "Failed to initialize GLEW" << std::endl;
        }
        CreateCamera({ 0.0f, 0.0f, 3.0f },{ 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, -1.0f });
        /*CreateCamera(glm::vec3({ 0.0f, -1.0f, 3.0f }), glm::vec3({ 0.0f, 1.0f, 0.0f }),
            glm::vec3({ 0.0f, 5.0f, -3.0f }));*/
        glEnable(GL_DEPTH_TEST);
    }

    void Render::CreateCamera(glm::vec3 position, glm::vec3 up, glm::vec3 front)
    {
        auto _camera = new Camera(position, up, front);
        this->m_camera = std::shared_ptr<Camera>(_camera);
    }


    std::vector<unsigned int> Render::addPlanetObject(const uint8_t* particles,
        int structSize,
        const int particleNum,
        unsigned int *planet_id,
        const char* vertexPath,
        const char* fragmentPath,
        const char* geometryPath)
    {
        Shader::ptr _shader = std::make_shared<Shader>(vertexPath, fragmentPath, geometryPath);
        this->m_shaders[ObjectType::Point].push_back(std::shared_ptr<Shader>(_shader));

        GLuint _VAO;
        glGenVertexArrays(1, &_VAO);
        glBindVertexArray(_VAO);
        std::vector<unsigned int> _VBOs(2);
        if (!m_VBOs.count(ObjectType::Point)){
            glGenBuffers(2, _VBOs.data());
            m_VBOs[ObjectType::Point] = _VBOs;
        }
        else{
            _VBOs = m_VBOs[ObjectType::Point];
        }
        m_planet_vertices.append(particles, particleNum * structSize);
        glBindBuffer(GL_ARRAY_BUFFER, _VBOs[0]);
        glBufferData(GL_ARRAY_BUFFER, m_planet_vertices.size(), (void*)m_planet_vertices.raw(), GL_DYNAMIC_DRAW);  // 允许动态修改
        glBindBuffer(GL_ARRAY_BUFFER, _VBOs[1]);
        glBufferData(GL_ARRAY_BUFFER, m_planet_vertices.size(), (void*)m_planet_vertices.raw(), GL_DYNAMIC_DRAW);
        m_planet_vertexs_counts[_VAO] = particleNum;

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, structSize, (GLvoid*)0);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, structSize, (GLvoid*)(8 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);

        this->m_planet_VAOs.push_back(_VAO);
        *planet_id = _VAO;
        return _VBOs;
    }

    void Render::addStarObject(const std::vector<float>& vertexs, const std::vector<unsigned int>& incides,
        unsigned int* star_id,
        const char* vertexPath, const char* fragmentPath,
        const char* geometryPath) {

        auto _shader = new Shader(vertexPath, fragmentPath, geometryPath);
        this->m_shaders[ObjectType::Triangle].push_back(std::shared_ptr<Shader>(_shader));

        GLuint _VAO, _VBO, _EBO;
        glGenVertexArrays(1, &_VAO);
        glGenBuffers(1, &_VBO);

        glBindVertexArray(_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, _VBO);
        glBufferData(GL_ARRAY_BUFFER, vertexs.size() * sizeof(float), vertexs.data(), GL_DYNAMIC_DRAW);  // 允许动态修改

        glGenBuffers(1, &_EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, incides.size() * sizeof(unsigned int), incides.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // 法线
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        // 纹理坐标
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        this->m_VBOs[ObjectType::Triangle].push_back(_VBO);
        this->m_star_VAOs.push_back(_VAO);
        this->m_star_EBOs.push_back(_EBO);
        this->m_star_vertexs_counts[_VAO] = incides.size();
        *star_id = _VAO;
    }

    void Render::addCubeObject(const char* vertexPath, char* fragmentPath,
        const char* texture_path, const char* texture_post_fix) {
        m_cube_shader = std::make_shared<Shader>(vertexPath, fragmentPath);

        float skyboxVertices[] = {
            // positions          
            -1.0f,  1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

            -1.0f,  1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
             1.0f, -1.0f,  1.0f
        };
        m_use_cube_texture = true;
        glGenVertexArrays(1, &m_cube_VAO);
        glGenBuffers(1, &m_cube_VBO);
        glBindVertexArray(m_cube_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_cube_VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

        addCubeTexture(texture_path, texture_post_fix);
        glEnable(GL_DEPTH_TEST);
        m_cube_shader->use();
        m_cube_shader->setInt("skybox", 0);
    }

    void Render::addStarTexture(const char* path) {
        m_star_texture = std::make_shared<Texture2D>(path);
        this->m_use_texture = true;
    }

    void Render::addCubeTexture(const char* path, const char* postFix) {
        m_cube_texture = std::make_shared<TextureCube>(path, postFix);
        this->m_use_texture = true;
    }


    void Render::update()
    {
        
    }

    void Render::cleanup(){
        for (int i = 0; i < this->m_star_VAOs.size(); i++){
            glDeleteVertexArrays(1, &this->m_star_VAOs[i]);
        }
        glDeleteBuffers(m_VBOs[ObjectType::Triangle].size(), m_VBOs[ObjectType::Triangle].data());
        glDeleteBuffers(m_star_EBOs.size(), m_star_EBOs.data());

        for (int i = 0; i < this->m_planet_VAOs.size(); i++) {
            glDeleteVertexArrays(1, &this->m_star_VAOs[i]);
        }
        glDeleteBuffers(m_VBOs[ObjectType::Point].size(), m_VBOs[ObjectType::Triangle].data());

        m_VBOs.clear();
        m_planet_VAOs.clear();
        m_planet_vertexs_counts.clear();
        m_planet_vertices.clear();

        m_star_vertexs_counts.clear();
        m_star_VAOs.clear();
        m_star_EBOs.clear();
    }

    void Render::render(unsigned int current_vbo){
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //render star
        for (size_t i = 0; i < m_star_VAOs.size(); i++){
            this->m_shaders[ObjectType::Triangle][i]->use();
            glBindVertexArray(m_star_VAOs[i]);
            if (m_use_texture) {
                m_star_texture->bind(0);
            }
            glDrawElements(GL_TRIANGLES, m_star_vertexs_counts[m_star_VAOs[i]], GL_UNSIGNED_INT, 0);
            if (m_use_texture) {
                m_star_texture->unBind();
            }
        }
        for (size_t i = 0; i < m_planet_VAOs.size(); i++){
            this->m_shaders[ObjectType::Point][i]->use();
            glBindVertexArray(m_planet_VAOs[i]);
            glPointSize(3.0f);
            glDrawArrays(GL_POINTS, 1728 * i, m_planet_vertexs_counts[m_planet_VAOs[i]]);
            GLenum error = glGetError();
            error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "OpenGL Error: " << error << std::endl;
            }
        }
        glBindVertexArray(0);
        // draw skybox as last
        if (m_use_cube_texture){
            glDepthFunc(GL_LEQUAL);  // change depth function so depth test passes when values are equal to depth buffer's content
            // skybox cube
            m_cube_shader->use();
            glBindVertexArray(m_cube_VAO);
            glActiveTexture(GL_TEXTURE0);
            m_cube_texture->bind(0);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);
            glDepthFunc(GL_LESS); // set depth function back to default
        }
        glfwSwapBuffers(this->window);
        glfwPollEvents();
    }

    void Render::SaveFrameBufferToImage(int frame_id, std::string path) {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glFinish(); // ← 加上这一句，确保渲染结束再读像素！！！

        std::vector<unsigned char> pixels(width * height * 3); // RGB
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        std::vector<unsigned char> flipped(pixels.size());
        for (int y = 0; y < height; ++y) {
            memcpy(&flipped[y * width * 3],
                &pixels[(height - y - 1) * width * 3],
                width * 3);
        }

        // 保存为 PNG，推荐用 stb_image_write 或其他库
        char buf[256];
        sprintf(buf, "frame_%04d.jpg", frame_id);
        std::string filename = path + std::string(buf);
        
        stbi_write_jpg(filename.c_str(), width, height, 3, flipped.data(), width * 3);
    }


    void Render::processInput(GLFWwindow* window)
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            this->m_camera->ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            this->m_camera->ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            this->m_camera->ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            this->m_camera->ProcessKeyboard(RIGHT, deltaTime);
            
    }

    void Render::framebuffer_size_callback(GLFWwindow* window, int width, int height)
    {
        glViewport(0, 0, width, height);
    }

    void Render::mouse_callback(GLFWwindow* window, double xposIn, double yposIn){
        static bool firstMouse = true;
        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        if (firstMouse){
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }
        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;
        //GetInstance()->m_camera->ProcessMouseMovement(xoffset, yoffset);
    }

    void Render::scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
        GetInstance()->m_camera->ProcessMouseScroll(static_cast<float>(yoffset));
    }

    bool Render::shouldClose() { return glfwWindowShouldClose(window); }
    

}