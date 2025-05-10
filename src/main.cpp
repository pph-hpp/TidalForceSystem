#include "TidalSystem.h"
#include "Render/Render.h"
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>
#include <cuda_runtime.h>


void executeV2() {
    Render::Render::ptr render = Render::Render::GetInstance();
    render->init(1400, 900);
    render->addStarTexture("../textures/sun_texture.jpg");

    TidalSystem::TidalSystem tidal_system(32);
    /*float angle_velocity = 1.2f * sqrt(G * STAR_MASS / pow(PLANET_STAR_DISTANCE, 3));*/
    float angle_velocity = 0.0f * sqrt(G * STAR_MASS / pow(PLANET_STAR_DISTANCE, 3));
    float velocity = 0.96f * sqrt(G * STAR_MASS / PLANET_STAR_DISTANCE);
    std::cout << "Planet angle velocity : " << angle_velocity << std::endl;

    /*tidal_system.addObject(TidalSystem::PlanetType::Planet, { STAR_POSITION.x + 400, STAR_POSITION.y + 300, 0 },
        PLANET_RADIUS, { -velocity, 0, 0 }, 14, {0.0f, 0.0f, angle_velocity });*/

    tidal_system.addObject(TidalSystem::PlanetType::Planet, { PLANET_STAR_DISTANCE, 0, 0 },
        PLANET_RADIUS, { 0, velocity, 0 }, 14, { 0.0f, 0.0f, angle_velocity });

    tidal_system.addObject(TidalSystem::PlanetType::Star, STAR_POSITION,
        BLOCK_HOLE_RADIUS, {0, 0, 0}, 100);
    std::cout << "Particle size: " << sizeof(Particle) << std::endl;
    render->addCubeObject("../glsl/skybox.vs", "../glsl/skybox.fs",
        "../textures/skybox/", ".png");

    float scale_size = 1.0 / (300);
    glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
    model = glm::scale(model, glm::vec3(scale_size, scale_size, scale_size));
    Render::Shader::ptr _shader;
    int fram_id = 0;
    while (!render->shouldClose()) {
        glm::mat4 projection = glm::perspective(glm::radians(render->m_camera->Zoom),
            (float)render->scr_width / (float)render->scr_height, 0.1f, 100.0f);
        glm::mat4 view = render->m_camera->GetViewMatrix();
        for (const auto& pair : render->m_shaders) {
            for (const auto& shader : pair.second) {
                shader->use();
                shader->setMat4("projection", projection);
                shader->setMat4("view", view);
                shader->setMat4("model", model);
            }
        }
        if (render->m_use_cube_texture) {
            _shader = render->m_cube_shader;
            _shader->use();
            view = glm::mat4(glm::mat3(view)); // remove translation from the view matrix
            _shader->setMat4("view", view);
            _shader->setMat4("projection", projection);
        }
        render->processInput(render->window);
        render->render(tidal_system.getCurrentRead());

        //±£´æÃ¿Ò»Ö¡Í¼Ïñ
        render->SaveFrameBufferToImage(fram_id++, "E:/s/hpc/TidalForceSimulater/TidalForceSystem/images/");

        auto start = std::chrono::high_resolution_clock::now();
        tidal_system.simulate(0.5);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time : " << duration.count() << " ms" << std::endl;
        /*std::cout << "1 epoch end" << std::endl;*/
        //break;
    }
    render->cleanup();
    tidal_system.finalize();
}

int main(){
    executeV2();
    return 0;
}
