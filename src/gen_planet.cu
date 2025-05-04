#include "gen_planet.cuh"
#include "ForceCalc.cuh"
#include <vector_types.h>
#include <iostream>
#include <curand_kernel.h>

__device__ float3 uniformSampleInSphere(curandState* state, float radius, float3 center) {
    float u = curand_uniform(state);  // [0,1)
    float theta = 2.0f * PI * curand_uniform(state);  // [0, 2π)
    float phi = acosf(1.0f - 2.0f * curand_uniform(state));  // [0, π]
    float r = cbrtf(u) * radius;  // 注意立方根！否则不是体积均匀分布
    float x = r * sinf(phi) * cosf(theta);
    float y = r * sinf(phi) * sinf(theta);
    float z = r * cosf(phi);

    return make_float3(center.x + x, center.y + y, center.z + z);
}

__device__ float3 sphericalToCartesian(float r, float theta, float phi, float3 center) {
    float x = r * sinf(phi) * cosf(theta);
    float y = r * sinf(phi) * sinf(theta);
    float z = r * cosf(phi);
    return make_float3(x + center.x, y + center.y, z + center.z);
}



__global__ void generateSphereParticles(Particle* particles, int* validCount,
    int gridSize, float3 center, float radius) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= gridSize || idy >= gridSize || idz >= gridSize) return;

    // 计算点的坐标
    float step = (2.0f * radius) / gridSize;  // 计算步长
    float x = center.x - radius + idx * step;
    float y = center.y - radius + idy * step;
    float z = center.z - radius + idz * step;

    // 计算距离，判断是否在球内
    if ((x - center.x) * (x - center.x) +
        (y - center.y) * (y - center.y) +
        (z - center.z) * (z - center.z) <= radius * radius) {
        // 计算存储位置
        int pos = atomicAdd(validCount, 1);
        particles[pos] = { x, y, z };
        particles[pos].index = pos;
        particles[pos].velocity = { 0, 0, 0 };
        particles[pos].is_connect = true;
        float v_angle = sqrt(G * BLOCK_HOLE_DENSITY * 4 / 3 * PI * pow(BLOCK_HOLE_RADIUS, 3) / pow(PLANET_STAR_DISTANCE, 3));
        /*particles[pos].angle_velocity = v_angle;*/
        particles[pos].angle_velocity = v_angle;
        particles[pos].total_angle = 0;
        particles[pos].angle_theta = atan2f((y - center.y), (x - center.x));
        particles[pos].angle_phi = acosf((z - center.z) / sqrt(pow((x - center.x), 2) + pow((y - center.y), 2)
            + pow((z - center.z), 2)));
        particles[pos].is_swallowed = false;
    }
    __syncthreads();
}


__global__ void generateRandomSpherePoints(
    Particle* particles,
    int numPoints,
    float3 center,
    float radius,
    unsigned int planet_id,
    unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    curandState state;
    curand_init(seed, idx, 0, &state);
    // 体积均匀采样的半径
    float u = curand_uniform(&state);  // [0,1)
    float r = radius * cbrtf(u);
    // 均匀采样角度
    float theta = 2.0f * PI * curand_uniform(&state);       // 方位角 [0, 2π)
    float phi = asinf(2.0f * curand_uniform(&state) - 1.0f) + PI / 2; // 天顶角 [0, π]
    float3 pos = sphericalToCartesian(r, theta, phi, center);

    particles[idx].position = pos;
    particles[idx].index = idx;
    particles[idx].velocity = make_float3(0, 0, 0);
    particles[idx].is_connect = true;
    particles[idx].total_angle = 0;
    particles[idx].angle_theta = theta;
    particles[idx].angle_phi = phi;
    particles[idx].is_swallowed = false;
    particles[idx].radius = r;
    particles[idx].mass = PARTICLE_DENSITY * 4 / 3 * PI * pow(PARTICLE_RADIUS, 3);
    particles[idx].planet_id = planet_id;
}


__global__ void generateDeterministicSpherePoints(
    Particle* particles,
    int numPoints,
    float3 center,
    float radius,
    unsigned int planet_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    // 推导合理分层数
    int radialDiv = cbrtf((float)numPoints);       // 半径方向层数
    int phiDiv = radialDiv * 2;                    // 天顶角划分
    int thetaDiv = radialDiv * 2;                  // 方位角划分
    int totalDiv = radialDiv * phiDiv * thetaDiv;
    if (idx >= totalDiv) return;  // 防止越界

    // 分层索引
    int r_idx = idx / (phiDiv * thetaDiv);
    int phi_idx = (idx / thetaDiv) % phiDiv;
    int theta_idx = idx % thetaDiv;
    // 均匀层体积 => 半径按立方根均匀分布
    float r = radius * cbrtf((float)(r_idx + 1) / (float)(radialDiv - 1));
    // 天顶角 ∈ [0, π]
    float phi = PI * (float)(phi_idx + 0.5f) / (float)phiDiv;
    // 方位角 ∈ [0, 2π]
    float theta = 2.0f * PI * (float)(theta_idx + 0.5f) / (float)thetaDiv;
    float3 pos = sphericalToCartesian(r, theta, phi, center);

    particles[idx].position = pos;
    particles[idx].index = idx;
    particles[idx].velocity = make_float3(0, 0, 0);
    particles[idx].is_connect = true;
    particles[idx].total_angle = 0;
    particles[idx].angle_theta = theta;
    particles[idx].angle_phi = phi;
    particles[idx].is_swallowed = false;
    particles[idx].radius = r;
    particles[idx].mass = PARTICLE_DENSITY * 4 / 3 * PI * pow(PARTICLE_RADIUS, 3);
    particles[idx].planet_id = planet_id;
}


void genPlanetDataHost(const int gridSize, float radius, float3 centerPos,
    std::vector<Particle>& particles, unsigned int planet_id, unsigned int& particleNum) {

    int totalPoints = gridSize * gridSize * gridSize;
    int maxParticles = totalPoints;

    // 分配 GPU 内存
    Particle* d_particles;
    int* d_validCount;
    cudaMalloc(&d_particles, maxParticles * sizeof(Particle));
    cudaMalloc(&d_validCount, sizeof(int));
    cudaMemset(d_validCount, -1, sizeof(int));

    // CUDA 线程块配置
    unsigned int blockSize1D = 256;
    unsigned int gridSize1D = (totalPoints + blockSize1D - 1) / blockSize1D;
    // 运行 CUDA 核函数
    generateRandomSpherePoints << <gridSize1D, blockSize1D >> > (
        d_particles,
        totalPoints,
        centerPos,
        radius,
        planet_id,
        time(NULL));
    /*generateDeterministicSpherePoints << <gridSize1D, blockSize1D >> > (
        d_particles,
        totalPoints,
        centerPos,
        radius,
        planet_id);*/
    cudaDeviceSynchronize();
    particleNum = totalPoints;
    std::cout << "particle num: " << particleNum << std::endl;
    // 拷贝数据回 CPU
    particles.resize(particleNum);
    cudaMemcpy(particles.data(), d_particles, particleNum * sizeof(Particle), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_particles);
    cudaFree(d_validCount);
}