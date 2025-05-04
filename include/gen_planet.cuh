#include "ParticleStruct.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

__global__ void generateSphereParticles(Particle* particles, int* validCount,
    int gridSize, float3 center, float radius);

__global__ void generateRandomSpherePoints(
    Particle* particles,
    int numPoints,
    float3 center,
    float radius,
    unsigned int planet_id,
    unsigned long seed);

__global__ void generateDeterministicSpherePoints(
    Particle* particles,
    int numPoints,
    float3 center,
    float radius);

void genPlanetDataHost(const int gridSize, float radius, float3 centerPos,
    std::vector<Particle>& particles, unsigned int planet_id, unsigned int& particleNum);