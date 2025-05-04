#include "gen_planet.cuh"
#include "ForceCalc.cuh"
#include <vector_types.h>
#include <iostream>
#include <curand_kernel.h>

__device__ float3 uniformSampleInSphere(curandState* state, float radius, float3 center) {
    float u = curand_uniform(state);  // [0,1)
    float theta = 2.0f * PI * curand_uniform(state);  // [0, 2��)
    float phi = acosf(1.0f - 2.0f * curand_uniform(state));  // [0, ��]
    float r = cbrtf(u) * radius;  // ע����������������������ȷֲ�
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

    // ����������
    float step = (2.0f * radius) / gridSize;  // ���㲽��
    float x = center.x - radius + idx * step;
    float y = center.y - radius + idy * step;
    float z = center.z - radius + idz * step;

    // ������룬�ж��Ƿ�������
    if ((x - center.x) * (x - center.x) +
        (y - center.y) * (y - center.y) +
        (z - center.z) * (z - center.z) <= radius * radius) {
        // ����洢λ��
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
    // ������Ȳ����İ뾶
    float u = curand_uniform(&state);  // [0,1)
    float r = radius * cbrtf(u);
    // ���Ȳ����Ƕ�
    float theta = 2.0f * PI * curand_uniform(&state);       // ��λ�� [0, 2��)
    float phi = asinf(2.0f * curand_uniform(&state) - 1.0f) + PI / 2; // �춥�� [0, ��]
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
    // �Ƶ�����ֲ���
    int radialDiv = cbrtf((float)numPoints);       // �뾶�������
    int phiDiv = radialDiv * 2;                    // �춥�ǻ���
    int thetaDiv = radialDiv * 2;                  // ��λ�ǻ���
    int totalDiv = radialDiv * phiDiv * thetaDiv;
    if (idx >= totalDiv) return;  // ��ֹԽ��

    // �ֲ�����
    int r_idx = idx / (phiDiv * thetaDiv);
    int phi_idx = (idx / thetaDiv) % phiDiv;
    int theta_idx = idx % thetaDiv;
    // ���Ȳ���� => �뾶�����������ȷֲ�
    float r = radius * cbrtf((float)(r_idx + 1) / (float)(radialDiv - 1));
    // �춥�� �� [0, ��]
    float phi = PI * (float)(phi_idx + 0.5f) / (float)phiDiv;
    // ��λ�� �� [0, 2��]
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

    // ���� GPU �ڴ�
    Particle* d_particles;
    int* d_validCount;
    cudaMalloc(&d_particles, maxParticles * sizeof(Particle));
    cudaMalloc(&d_validCount, sizeof(int));
    cudaMemset(d_validCount, -1, sizeof(int));

    // CUDA �߳̿�����
    unsigned int blockSize1D = 256;
    unsigned int gridSize1D = (totalPoints + blockSize1D - 1) / blockSize1D;
    // ���� CUDA �˺���
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
    // �������ݻ� CPU
    particles.resize(particleNum);
    cudaMemcpy(particles.data(), d_particles, particleNum * sizeof(Particle), cudaMemcpyDeviceToHost);

    // �ͷ� GPU �ڴ�
    cudaFree(d_particles);
    cudaFree(d_validCount);
}