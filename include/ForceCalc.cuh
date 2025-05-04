#include "ParticleStruct.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__host__ __device__ float calDistance3D(const float3& a, const float3& b);

__host__ __device__ float3 operator+(float3 a, float3 b);

__host__ __device__ float3 operator-(float3 a, float3 b);

__host__ __device__ float3 operator*(float s, float3 v);

__host__ __device__ float3 operator/(float3 s, float v);

__device__ bool swallowDetection(const float3* position);

__device__ void SwallowedByStar(Particle* p, const StarInfo* star_info, unsigned int star_num);

__host__ __device__ float calLengthOfVec(const float3& a);

__host__ __device__ float3 calGravity(const float3& p_target, const float3& p_source, const float mass1, const float mass2);

__host__ __device__ float3 calGravityV3(const float3& p_target, const float3& p_source, const float mass1, const float mass2,
    const float3& center_pos, const float radius);

__host__ __device__ float3 updateVec(const float3* vec,
    const float3* offset,
    const float deltaTime = 1);

__device__ float3 cross(const float3& r1, const float3& r2);

__device__ float dot(const float3& r1, const float3& r2);

__device__ float3 CalSpinCentrifugal(const float3& w, const float3& r);

__device__ float3 rotateRodrigues(float3 p, float3 v_spin, float delta_time);

__device__ int flatten2D(int i, int j, int num);

__device__ void CalTotalForce(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const float particle_mass,
    float3* totalForce);

__device__ void CalTotalForce_v2(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const float particle_mass,
    float3* totalForce);

__device__ inline void CalTotalForce_v3(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const float particle_mass,
    float3* totalForce);


__device__ void updateParticleRK4(Particle& p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const float particle_mass,
    float dt);

__device__ void updateCenterRK4(float3* position,
    float3* velocity,
    float dt);

__device__ void CalTotalForceV2(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const StarInfo* star_info,
    unsigned int star_num,
    float3* totalForce);


__device__ void CalTotalForceV3(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const StarInfo* star_info,
    const StarInfo* planet_info,
    unsigned int star_num,
    float3* totalForce);

__device__ void updateParticleRK4V2(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const StarInfo* star_info,
    const int star_num,
    float dt);


__device__ void updateParticleRK4V3(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const StarInfo* star_info,
    const StarInfo* planet_info,
    const int star_num,
    float dt);