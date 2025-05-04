#pragma once
#include <driver_types.h>
#include <string>
#include <memory>
#include "ParticleStruct.h"
//#include "Render/Render.h"
#include <vector>

namespace TidalSystem
{
	enum class PlanetType {
		Planet,  // 行星
		Star     // 恒星
	};

	class TidalSystem
	{
		unsigned int m_blockSize;
		unsigned int m_particle_num = 0;		//所有移动星体质点总数
		unsigned int m_planet_num = 0;		//移动星体数量
		unsigned int m_star_num = 0;

		//VAO ids
		std::vector<unsigned int> m_star_ids;
		std::vector<unsigned int> m_planet_ids;

		std::vector<StarInfo> m_planet_info;
		std::vector<StarInfo> m_star_info;
		StarInfo* m_planet_info_device[2];	//分别为host端和device端数据
		StarInfo* m_star_info_device[2];

		//为连接体提前分配内存，避免内存反复分配
		CenterStatus* connect_status;
		int* connect_id_map;

		::Particle* m_deviceParticles[2];	//opengl映射时使用，用于双缓冲，指向两个cuda数组

		std::vector<unsigned int> m_vbo_planets{ 0, 0 };	//用于双缓冲
		cudaGraphicsResource* m_pGRes[2];

	private:
		unsigned int m_current_read;
		unsigned int m_current_write;
		bool first_excute = true;

	public:
		typedef std::shared_ptr<TidalSystem> ptr;

		TidalSystem(unsigned int blockSize);

		void setBlockSize(unsigned int size);
		void setPosition(unsigned int vbo, float* data);
		bool genPlanetData(int gridSize, float radius, float3 centerPos, float3 CenterVel, float3 rotationVel);
		bool addObject(PlanetType type, float3 centerPos, float radius,
			float3 centerVel = { 0, 0, 0 }, int gridSize = 0, float3 rotationVel = { 0, 0, 0 });
		void generateSphere(int sectors, int stacks, float radius,
			float3 centerPos, std::vector<float>& vertices,
			std::vector<unsigned int>& indices);

		void loadDatFile(const std::string& path);

		void updateDeviceData();

		unsigned int& getBlockSize() { return m_blockSize; }
		unsigned int& getNumParticles() { return m_particle_num; }
		unsigned int getNumParticles()const { return m_particle_num; }
		unsigned int getCurrentRead() { return m_current_read; }
		float getPlanetMass(unsigned int id) { return m_planet_info[id].mass; }
		void setPlanetVelocity(unsigned int id, const float3& vel);

		void simulate(float deltaTime);

		void registerCudaGraphicsResource();

	public:
		void initialize(float v_angle);
		void finalize();

	};
}
