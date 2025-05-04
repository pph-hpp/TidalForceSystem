#include "TidalSystem.h"
//#define GLEW_STATIC
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include "MultiPlanet.cuh"
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <mutex>
#include "Render/render.h"
#include "CudaTools.cuh"
#include "gen_planet.cuh"

namespace TidalSystem
{
    TidalSystem::TidalSystem(unsigned int blockSize) : m_blockSize(blockSize),
        m_current_read(0), m_current_write(1)
    {
        m_blockSize = blockSize;
        m_particle_num = 0;
    }

    void TidalSystem::initialize(float v_angle) {

    }

    void TidalSystem::registerCudaGraphicsResource() {
        cudaGraphicsGLRegisterBuffer(&m_pGRes[0], m_vbo_planets[0], cudaGraphicsRegisterFlagsNone);
        cudaGraphicsGLRegisterBuffer(&m_pGRes[1], m_vbo_planets[1], cudaGraphicsRegisterFlagsNone);

        CHECK_CUDA_ERROR("TidalSystem::registerCudaGraphicsResource failed.");
    }

    void TidalSystem::setPosition(unsigned int vbo, float* data) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(data), data);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        CHECK_CUDA_ERROR("TidalSystem::setPosition failed.");
    }

    void TidalSystem::generateSphere(int sectors, int stacks, float radius,
        float3 centerPos, std::vector<float>& vertices,
        std::vector<unsigned int>& indices) {
        vertices.clear();
        indices.clear();

        // 生成顶点数据
        for (int i = 0; i <= stacks; ++i) {
            float phi = PI * i / stacks;  // 纬度角 [0, π]
            float v = 1.0f - (float)i / stacks;  // 纹理坐标 V

            for (int j = 0; j <= sectors; ++j) {
                float theta = 2.0f * PI * j / sectors;  // 经度角 [0, 2π]
                float u = (float)j / sectors;  // 纹理坐标 U

                // 球面坐标计算
                float x = sin(phi) * cos(theta);
                float y = cos(phi);
                float z = sin(phi) * sin(theta);

                // 顶点坐标
                vertices.push_back(radius * x + centerPos.x);
                vertices.push_back(radius * y + centerPos.y);
                vertices.push_back(radius * z + centerPos.z);

                // 法向量（单位球上法线与位置相同）
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);

                // 纹理坐标 (u, v)
                vertices.push_back(u);
                vertices.push_back(v);
            }
        }
        // 生成索引数据（GL_TRIANGLES 方式）
        for (int i = 0; i < stacks; ++i) {
            for (int j = 0; j < sectors; ++j) {
                int first = i * (sectors + 1) + j;
                int second = first + sectors + 1;

                indices.push_back(first);
                indices.push_back(second);
                indices.push_back(first + 1);

                indices.push_back(second);
                indices.push_back(second + 1);
                indices.push_back(first + 1);
            }
        }
    }

    bool TidalSystem::genPlanetData(int gridSize, float radius, float3 centerPos, float3 CenterVel,
        float3 rotationVel) 
    {
        std::vector<Particle> _pArray;
        unsigned int _pNnum = 0;
        genPlanetDataHost(gridSize, radius, centerPos, _pArray, m_planet_num, _pNnum);

        if (_pArray.size() != _pNnum) return false;
        void* ptr = static_cast<void*>(_pArray.data());
        unsigned int planet_id;
        m_vbo_planets = Render::Render::GetInstance()->addPlanetObject(reinterpret_cast<uint8_t*>(_pArray.data()),
            sizeof(Particle), _pNnum, &planet_id, "../glsl/particle.vs", "../glsl/particle.fs");
        float mass = _pNnum * PARTICLE_DENSITY * 4 / 3 * PI * pow(PARTICLE_RADIUS, 3);
        m_planet_ids.push_back(planet_id);
        m_planet_info.push_back(StarInfo{ centerPos, CenterVel, rotationVel, mass,
            m_particle_num, _pNnum, m_planet_num, radius });
        m_particle_num += _pNnum;
        m_planet_num++;
        return true;
    }


    bool TidalSystem::addObject(PlanetType type, float3 centerPos,
        float radius, float3 centerVel, int gridSize, float3 rotationVel) {
        if (type == PlanetType::Planet) {
            genPlanetData(gridSize, radius, centerPos, centerVel, rotationVel);
            registerCudaGraphicsResource();
        }
        else {
            std::vector<float> _vertexs;
            std::vector<unsigned int> _indices;
            generateSphere(gridSize, gridSize, radius, centerPos, _vertexs, _indices);
            unsigned int star_id;
            Render::Render::GetInstance()->addStarObject(_vertexs, _indices, &star_id, "../glsl/star.vs", "../glsl/star.fs");
            m_star_ids.push_back(star_id);
            float mass = STAR_MASS;
            m_star_info.push_back(StarInfo{ centerPos, centerVel,{0.0f, 0.0f, 0.0f}, mass, 0, 0 });
            m_star_num++;
        }
        return true;
    }

    void TidalSystem::loadDatFile(const std::string& path) {

    }

    void TidalSystem::updateDeviceData() {
        cudaMalloc((void**)&m_planet_info_device[1], sizeof(StarInfo) * m_planet_num);
        cudaMalloc((void**)&m_star_info_device[1], sizeof(StarInfo) * m_star_num);
        cudaMalloc((void**)&connect_status, sizeof(CenterStatus) * m_planet_num);
        cudaMalloc((void**)&connect_id_map, sizeof(int) * m_planet_num);

        m_planet_info_device[0] = m_planet_info.data();
        m_star_info_device[0] = m_star_info.data();
        cudaMemcpy(m_planet_info_device[1], m_planet_info.data(), sizeof(StarInfo) * m_planet_num,
            cudaMemcpyHostToDevice);
        cudaMemcpy(m_star_info_device[1], m_star_info.data(), sizeof(StarInfo) * m_star_num,
            cudaMemcpyHostToDevice);
    }

    void TidalSystem::simulate(float deltaTime) {
        if (first_excute){
            first_excute = false;
            updateDeviceData();
        }
        TidalForceV2Host(m_deviceParticles, m_pGRes, m_current_read,
            m_star_info_device, m_planet_info_device,
            connect_status, connect_id_map,
            m_star_num, m_planet_num,
            m_particle_num, deltaTime, m_blockSize);

        static int execution_times = 0;
        std::cout << "Execution times: " << ++execution_times << std::endl;
        //CHECK_CUDA_ERROR("TidalSystem::simulate failed.");
        std::swap(m_current_read, m_current_write);
    }

    void TidalSystem::setPlanetVelocity(unsigned int id, const float3& vel) {
        if (id >= m_planet_info.size()){
            std::cout << "There is not planet " << id << std::endl;
            return;
        }
        this->m_planet_info[id].vel = vel;
    }

    void TidalSystem::finalize() {
        cudaGraphicsUnregisterResource(m_pGRes[0]);
        cudaGraphicsUnregisterResource(m_pGRes[1]);
        glDeleteBuffers(2, m_vbo_planets.data());

        CHECK_CUDA_ERROR("TidalSystem::finalize failed.");
    }

    void TidalSystem::setBlockSize(unsigned int size) {
        this->m_blockSize = size;
    }
}
