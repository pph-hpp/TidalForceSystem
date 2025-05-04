#include "ForceCalc.cuh"
#include "gen_planet.cuh"
#include <vector_types.h>
#include <iostream>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <vector>
#include <unordered_map>
#include <numeric>



//随质心移动方式
__global__ void UpdateParticleStatusDevice(Particle* p_read,   //粒子状态双缓冲
    Particle* p_write,  //粒子状态双缓冲
    const CenterStatus* connect_status,
    const int* connect_id_map,
    const StarInfo* star_info,
    const unsigned int star_num,
    const StarInfo* planet_info,
    const unsigned int ParticleNum,
    const float delta_time)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ParticleNum) {
        return;
    }
    else{
        Particle* p = &p_read[tid];
        Particle* p_new = &p_write[tid];
        p_new->position = p->position;
        p_new->velocity = p->velocity;
        p_new->is_swallowed = p->is_swallowed;
        p_new->is_connect = p->is_connect;
        if (p->is_swallowed) {
            p_new->is_swallowed = true;
            p_new->position = STAR_POS;
        }
        else {
            updateParticleRK4V2(p_new, p_read, blockDim.x, ParticleNum, star_info, star_num, delta_time);
            if (p->is_connect){
                //需引入离心力，判断是否达到断裂极限
                float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);
                //计算所有力，包括恒星对粒子的引力
                CalTotalForceV2(p, p_read, blockDim.x, ParticleNum, star_info, star_num, &totalForce);
                float3 judge_force = totalForce;
                //由加速度产生的撕扯作用
                int _id = connect_id_map[p->planet_id];
                judge_force = judge_force - p->mass * connect_status[_id].acceleration;

                float f_n = judge_force.x * sin(p->angle_phi) * cos(p->angle_theta) +
                    judge_force.y * sin(p->angle_phi) * sin(p->angle_theta) +
                    judge_force.z * cos(p->angle_phi);
                float f_t = sqrt(judge_force.x * judge_force.x
                    + judge_force.y * judge_force.y
                    + judge_force.z * judge_force.z - f_n * f_n);
                float _area = 4 * PI * PARTICLE_RADIUS * PARTICLE_RADIUS;
                float p_n = f_n / _area;
                float p_t = f_t / _area;

                float3 _offset = { p->radius * sin(p->angle_phi) * cos(p->angle_theta),
                    p->radius * sin(p->angle_phi) * sin(p->angle_theta),
                    p->radius * cos(p->angle_phi) };
                p_new->position = planet_info[p->planet_id].pos + _offset;
                p_new->velocity = planet_info[p->planet_id].vel;
                //应允许外半球也被吸入，内半球也允许被甩出。本身强度应能够抗衡引力塌缩
                if (abs(p_n) >= TENSILE_STRENGTH || p_t >= SHEAR_STRENGTH) {
                    p_new->is_connect = false;
                    p->is_connect = false;
                }
            }
            if (star_num > 0 && !p->is_swallowed){
                SwallowedByStar(p_new, star_info, star_num);
            }
        }
    }
}


__global__ void UpdateParticleStatusDeviceWithRotation(Particle* p_read,   //粒子状态双缓冲
    Particle* p_write,  //粒子状态双缓冲
    const CenterStatus* connect_status,
    const int* connect_id_map,
    const StarInfo* star_info,
    const unsigned int star_num,
    const StarInfo* planet_info,
    const unsigned int ParticleNum,
    const unsigned int tileSize,
    const float delta_time)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ParticleNum) {
        return;
    }
    else {
        Particle p = p_read[tid];
        Particle p_new = p;
        //Particle* p_new = &p_write[tid];
        if (p.is_swallowed) {
            p_new.is_swallowed = true;
            p_new.position = STAR_POS;
        }
        else {
            updateParticleRK4V2(&p_new, p_read, tileSize, ParticleNum, star_info, star_num, delta_time);
            if (p.is_connect) {
                float r = planet_info[p.planet_id].radius;
                float3 v_spin = planet_info[p.planet_id].v_spin;
                p_new.angle_theta = fmodf(p.angle_theta + v_spin.z * delta_time, 2 * PI);
                float3 pos_vec = { sin(p.angle_phi) * cos(p_new.angle_theta),
                    sin(p.angle_phi) * sin(p_new.angle_theta),
                    cos(p.angle_phi) };
                float3 _offset = p.radius * pos_vec;
                p_new.position = planet_info[p.planet_id].pos + _offset;
                p_new.velocity = planet_info[p.planet_id].vel;
                //需引入离心力，判断是否达到断裂极限
                float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);
                //计算所有力，包括恒星对粒子的引力
                CalTotalForceV2(&p, p_read, tileSize, ParticleNum, star_info, star_num, &totalForce);
                float3 judge_force = totalForce;
                //由加速度产生的离心作用
                int _id = connect_id_map[p.planet_id];
                judge_force = judge_force - p.mass * connect_status[_id].acceleration;

                //由自转产生的离心作用
                float3 v_rotation = cross(v_spin, _offset);
                float3 a_rotation = cross(v_spin, v_rotation);
                judge_force = judge_force - p.mass * a_rotation;

                float f_n = dot(judge_force, pos_vec);
                float f_t = sqrt(judge_force.x * judge_force.x
                    + judge_force.y * judge_force.y
                    + judge_force.z * judge_force.z - f_n * f_n);
                float _area = 4 * PI * PARTICLE_RADIUS * PARTICLE_RADIUS;
                f_n = f_n / _area;
                f_t = f_t / _area;

                if (abs(f_n) >= TENSILE_STRENGTH || f_t >= SHEAR_STRENGTH) {
                    p_new.is_connect = false;
                    p_new.velocity = p_new.velocity + v_rotation;
                }
            }
            if (star_num > 0 && !p.is_swallowed) {
                SwallowedByStar(&p_new, star_info, star_num);
            }
        }
        p_write[tid] = p_new;
    }
}


__global__ void UpdateParticleStatusDeviceSpring(Particle* p_read,   //粒子状态双缓冲
    Particle* p_write,  //粒子状态双缓冲
    const CenterStatus* connect_status,
    const int* connect_id_map,
    const StarInfo* star_info,
    const unsigned int star_num,
    const StarInfo* planet_info,
    const unsigned int ParticleNum,
    const float delta_time)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ParticleNum) {
        return;
    }
    else {
        Particle* p = &p_read[tid];
        Particle* p_new = &p_write[tid];

        float r = planet_info[p->planet_id].radius;
        float3 v_spin = planet_info[p->planet_id].v_spin;
        p_new->angle_theta = fmodf(p->angle_theta + v_spin.z * delta_time, 2 * PI);
        float3 _offset = { p->radius * sin(p->angle_phi) * cos(p->angle_theta),
            p->radius * sin(p->angle_phi) * sin(p->angle_theta),
            p->radius * cos(p->angle_phi) };
        p_new->position = planet_info[p->planet_id].pos + _offset;
        p_new->velocity = planet_info[p->planet_id].vel;
        //需引入离心力，判断是否达到断裂极限
        float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);
        //计算所有力，包括恒星对粒子的引力
        CalTotalForceV2(p, p_read, blockDim.x, ParticleNum, star_info, star_num, &totalForce);
        float3 judge_force = totalForce;
        
        //由加速度产生的离心作用
        int _id = connect_id_map[p->planet_id];
        judge_force = judge_force - p->mass* connect_status[_id].acceleration;

        //由自转产生的离心作用
        StarInfo _info = planet_info[p->planet_id];
        float3 r_rotation = { r * sinf(p->angle_phi) * cosf(p->angle_theta),
            r * sinf(p->angle_phi) * sinf(p->angle_theta),
            r * cosf(p->angle_phi) };
        float3 v_rotation = cross(v_spin, r_rotation);
        float3 a_rotation = cross(v_spin, v_rotation);
        judge_force = judge_force - p->mass * a_rotation;

        /*float3 _n = { sin(p->angle_phi) * cos(p->angle_theta),
            sin(p->angle_phi) * sin(p->angle_theta),
            cos(p->angle_phi) };
        float f_n = dot(judge_force, _n);
        float3 offset_n = (f_n * _n) / k_spring_n;
        float3 offset_t = (judge_force - f_n * _n) / k_spring_t;
        p_new->position = p_new->position + offset_n + offset_t;*/
        p_new->position = p_new->position + (judge_force / k_spring / p->mass);
    }
}

//计算质心位置
__global__ void CalcCentroidDevice(Particle* p_read,
    float3* blockSums, //质心位置，待写入
    float* blockMass, //质心质量，待写入
    const unsigned int ParticleNum) 
{
    extern __shared__ float4 s_mem[];
    float3* s_pos = (float3*)s_mem;              // 前 blockDim.x * sizeof(float3)
    float* s_mass = (float*)&s_pos[blockDim.x];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    s_pos[tid] = (idx < ParticleNum)
        ? (p_read[idx].is_connect ? p_read[idx].mass * p_read[idx].position : make_float3(0.0f, 0.0f, 0.0f))
        : make_float3(0.0f, 0.0f, 0.0f);
    s_mass[tid] = (idx < ParticleNum)
        ? (p_read[idx].is_connect ? p_read[idx].mass : 0.0f)
        : 0.0f;
    __syncthreads();
    for (unsigned int gap = blockDim.x / 2 ;gap > 0; gap >>= 1){
        if (tid < gap){
            s_pos[tid] = s_pos[tid] + s_pos[tid + gap];
            s_mass[tid] = s_mass[tid] + s_mass[tid + gap];
        }
        __syncthreads();
    }
    if (tid == 0){
        blockSums[blockIdx.x] = s_pos[0];
        blockMass[blockIdx.x] = s_mass[0];
    }
}


//计算质心加速度
//只有块体质量不为0时调用
//即：planet_info->mass != 0
//计算所有行星质点对几个块体的加速度，只计算零散粒子的引力作用
__global__ void CalcCenterStatusDevice(
    Particle* particles,
    CenterStatus* connect_planets,
    float3* center_acc,
    const unsigned int connect_planet_num,
    const unsigned int ParticleNum,
    const unsigned int blockSize)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float3 s_acc[];

#pragma unroll
    for (unsigned int i = 0; i < connect_planet_num; i++){
        if (idx >= ParticleNum || (particles[idx].is_connect && particles[idx].planet_id == connect_planets[i].id)){
            s_acc[flatten2D(i, tid, blockDim.x)] = make_float3(0.0f, 0.0f, 0.0f);
        }
        else{
            float3 _force = make_float3(0.0f, 0.0f, 0.0f);
            float3 _pos = particles[idx].position;
            float _mass = particles[idx].mass;
            _force = calGravity(connect_planets[i].pos, _pos, connect_planets[i].mass, _mass);
            s_acc[flatten2D(i, tid, blockDim.x)] = _force / connect_planets[i].mass;
        }
    }
#pragma unroll
    for (unsigned int gap = blockDim.x / 2; gap > 0; gap >>= 1) {
        for (unsigned int i = 0; i < connect_planet_num; i++) {
            if (tid < gap) {
                s_acc[flatten2D(i, tid, blockDim.x)] = s_acc[flatten2D(i, tid, blockDim.x)]
                    + s_acc[flatten2D(i, tid + gap, blockDim.x)];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (unsigned int i = 0; i < connect_planet_num; i++) {
            center_acc[i * gridDim.x + blockIdx.x] = s_acc[flatten2D(i, 0, blockDim.x)];
        }
    }
}

void UpdateCentriod(
    Particle* particles,
    StarInfo* planet_info,
    const unsigned int planet_num,
    const unsigned int blockSize,
    std::vector<CenterStatus>& connect_planets,
    std::vector<int>& connected_id_map)
{
    float3* blockPos;
    float* blockMass;
    std::vector<float3> blockPosVec;
    std::vector<float> blockMassVec;
    connect_planets.clear();
    connected_id_map.clear();
    connected_id_map.resize(planet_num);

    //int blockSize, minGridSize;
    //int sharedMemPerBlock = 512 * sizeof(float4);  // 估计初始值
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, CalcCentroidDevice, sharedMemPerBlock, 0);
    
    for (unsigned int i = 0; i < planet_num; i++){
        unsigned int blockNum = (planet_info[i].particle_num + blockSize - 1) / blockSize;
        cudaMalloc(&blockPos, blockNum * sizeof(float3));
        cudaMalloc(&blockMass, blockNum * sizeof(float));
        
        Particle* p_read = particles + planet_info[i].offset;
        
        CalcCentroidDevice<<<blockNum, blockSize, sizeof(float4) * blockSize>>>(p_read, blockPos, blockMass, planet_info[i].particle_num);
        cudaDeviceSynchronize();
        blockPosVec.resize(blockNum);
        blockMassVec.resize(blockNum);
        cudaMemcpy(blockPosVec.data(), blockPos, blockNum * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(blockMassVec.data(), blockMass, blockNum * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(blockPos);
        cudaFree(blockMass);

        float mass = std::accumulate(blockMassVec.begin(), blockMassVec.end(), 0.0f);
        if (mass <= 1e-7){
            continue;
        }
        float3 pos_sum = make_float3(0.0f, 0.0f, 0.0f);
        for (unsigned int j = 0; j < blockNum; j++){
            pos_sum = pos_sum + blockPosVec[j];
        }
        float3 center_pos = pos_sum / mass;
        CenterStatus _status{ center_pos, {0, 0, 0}, mass, planet_info[i].id };
        connected_id_map[planet_info[i].id] = connect_planets.size();
        connect_planets.push_back(_status);
    }
}

void UpdateBlockStatus(
    Particle* particles,
    unsigned int ParticleNum,
    std::vector<CenterStatus>& connect_planets,
    StarInfo* star_info,
    const unsigned int star_num,
    const unsigned int blockSize)
{
    unsigned int blockNum = (ParticleNum + blockSize - 1) / blockSize;
    float3* center_acc;
    cudaMalloc((void**)&center_acc, connect_planets.size() * sizeof(float3) * blockNum);
    CenterStatus* connect_planets_device;
    cudaMalloc((void**)&connect_planets_device, sizeof(CenterStatus) * connect_planets.size());
    cudaMemcpy(connect_planets_device, connect_planets.data(), sizeof(CenterStatus) * connect_planets.size(),
        cudaMemcpyHostToDevice);

    unsigned int _num = connect_planets.size();
    //计算非此连接体的所有行星物质对该连接体的加速度
    CalcCenterStatusDevice<<<blockNum, blockSize, sizeof(float3) * blockSize * _num>>>(particles,
        connect_planets_device,
        center_acc,
        _num,
        ParticleNum,
        blockSize);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    std::vector<float3> h_acc(blockNum * connect_planets.size());
    cudaMemcpy(h_acc.data(), center_acc, connect_planets.size() * blockNum * sizeof(float3),
        cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < _num; i++){
        auto begin = h_acc.begin() + i * connect_planets.size();
        connect_planets[i].acceleration = std::accumulate(begin,
            begin + blockNum, make_float3(0.0f, 0.0f, 0.0f));
    }
    cudaFree(center_acc);
    cudaFree(connect_planets_device);
    h_acc.clear();

    //计算恒星对连接体产生的引力/加速度
    for (unsigned int i = 0; i < connect_planets.size(); i++){
        for (unsigned int k = 0; k < star_num; k++){
            float3 _force = calGravity(connect_planets[i].pos, star_info[k].pos, connect_planets[i].mass, star_info[k].mass);
            connect_planets[i].acceleration = connect_planets[i].acceleration + _force / connect_planets[i].mass;
        }
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void UpdateBlockRK4Step(
    Particle* particles,
    unsigned int ParticleNum,
    std::vector<CenterStatus>& connect_planets,
    CenterStatus* connect_planets_device,
    std::vector<float3>& h_acc,
    float3* center_acc,
    StarInfo* star_info,
    const unsigned int star_num,
    unsigned int blockNum,
    const unsigned int blockSize) 
{
    float _num = connect_planets.size();
    //int minGridSize = 0;
    //int blockSize = 0;
    //int initialEstimate = 512 * sizeof(float3);  // 粗略估值
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, CalcCenterStatusDevice, initialEstimate, 0);
    //unsigned int blockNum = (ParticleNum + blockSize - 1) / blockSize;
    CalcCenterStatusDevice << <blockNum, blockSize, sizeof(float3)* blockSize* _num >> > (particles,
        connect_planets_device,
        center_acc,
        _num,
        ParticleNum,
        blockSize);
    cudaDeviceSynchronize();
    cudaMemcpy(h_acc.data(), center_acc, connect_planets.size() * blockNum * sizeof(float3),
        cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < _num; i++) {
        auto begin = h_acc.begin() + i * blockNum;
        connect_planets[i].acceleration = std::accumulate(begin,
            begin + blockNum, make_float3(0.0f, 0.0f, 0.0f));
        for (unsigned int k = 0; k < star_num; k++) {
            float3 _force = calGravity(connect_planets[i].pos, star_info[k].pos, connect_planets[i].mass, star_info[k].mass);
            connect_planets[i].acceleration = connect_planets[i].acceleration + _force / connect_planets[i].mass;
        }
    }
}


void UpdateBlockStatusRK4(
    Particle* particles,
    unsigned int ParticleNum,
    std::vector<CenterStatus>& connect_planets,
    std::vector<float3> &v0,
    StarInfo* star_info,
    const unsigned int star_num,
    float delta_time,
    const unsigned int blockSize)
{
    unsigned int blockNum = (ParticleNum + blockSize - 1) / blockSize;
    float3* center_acc;
    cudaMalloc((void**)&center_acc, connect_planets.size() * sizeof(float3) * blockNum);
    CenterStatus* connect_planets_device;
    cudaMalloc((void**)&connect_planets_device, sizeof(CenterStatus) * connect_planets.size());
    cudaMemcpy(connect_planets_device, connect_planets.data(), sizeof(CenterStatus) * connect_planets.size(),
        cudaMemcpyHostToDevice);
    std::vector<float3> h_acc(blockNum * connect_planets.size());
    unsigned int _num = connect_planets.size();
    std::vector<float3> k1_v(_num);
    std::vector<float3> k1_p(_num);
    std::vector<float3> k2_v(_num);
    std::vector<float3> k2_p(_num);
    std::vector<float3> p0(_num);

    UpdateBlockRK4Step(particles, ParticleNum, connect_planets, connect_planets_device, h_acc, 
        center_acc, star_info, star_num, blockNum, blockSize);
    for (unsigned int i = 0; i < _num; i++) {
        k1_v[i] = connect_planets[i].acceleration;
        k1_p[i] = v0[i];
        k2_v[i] = k1_v[i];
        k2_p[i] = v0[i] + 0.5f * delta_time * k1_v[i];
        p0[i] = connect_planets[i].pos;
        connect_planets[i].pos = p0[i] + 0.5f * delta_time * k2_p[i];
    }

    //RK3th Step
    std::vector<float3> k3_v(_num);
    std::vector<float3> k3_p(_num);
    cudaMemcpy(connect_planets_device, connect_planets.data(), sizeof(CenterStatus) * _num,
        cudaMemcpyHostToDevice);
    UpdateBlockRK4Step(particles, ParticleNum, connect_planets, connect_planets_device, h_acc,
        center_acc, star_info, star_num, blockNum, blockSize);
    for (unsigned int i = 0; i < _num; i++) {
        k3_v[i] = connect_planets[i].acceleration;
        k3_p[i] = v0[i] + 0.5f * delta_time * k2_v[i];
        connect_planets[i].pos = p0[i] + delta_time * k3_p[i];
    }

    //RK4th Step
    std::vector<float3> k4_v(_num);
    std::vector<float3> k4_p(_num);
    cudaMemcpy(connect_planets_device, connect_planets.data(), sizeof(CenterStatus) * _num,
        cudaMemcpyHostToDevice);
    UpdateBlockRK4Step(particles, ParticleNum, connect_planets, connect_planets_device, h_acc,
        center_acc, star_info, star_num, blockNum, blockSize);

    for (unsigned int i = 0; i < _num; i++) {
        k4_v[i] = connect_planets[i].acceleration;
        k4_p[i] = v0[i] + delta_time * k3_v[i];
        connect_planets[i].pos = p0[i] + (delta_time / 6.0f) * (k1_p[i] + 2.0f * k2_p[i] + 2.0f * k3_p[i] + k4_p[i]);
        v0[i] = v0[i] + (delta_time / 6.0f) * (k1_v[i] + 2.0f * k2_v[i] + 2.0f * k3_v[i] + k4_v[i]);
    }

    cudaFree(center_acc);
    cudaFree(connect_planets_device);
    h_acc.clear();
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}


void UpdateBlockStatusRK4(
    StarInfo* planet_info,
    const unsigned int planet_num,
    StarInfo* star_info,
    const unsigned int star_num,
    float dt)
{
    for (size_t i = 0; i < planet_num; i++){
        float3 p0 = planet_info[i].pos;
        float3 v0 = planet_info[i].vel;
        float3 a0 = make_float3(0.0f, 0.0f, 0.0f);
        for (size_t j = 0; j < star_num; j++){
            a0 = a0 + calGravity(p0, star_info[j].pos, planet_info[i].mass, star_info[j].mass);
        }
        a0 = a0 / planet_info[i].mass;

        float3 k1_v = a0;
        float3 k1_p = v0;

        StarInfo tmp2 = planet_info[i];
        float3 k2_v = make_float3(0.0f, 0.0f, 0.0f);
        for (size_t j = 0; j < star_num; j++) {
            k2_v = k2_v + calGravity(tmp2.pos, star_info[j].pos, tmp2.mass, star_info[j].mass);
        }
        k2_v = k2_v / tmp2.mass;
        float3 k2_p = v0 + 0.5f * dt * k1_v;

        StarInfo tmp3;
        tmp3.pos = p0 + 0.5f * dt * k2_p;
        tmp3.vel = v0 + 0.5f * dt * k2_v;
        tmp3.mass = planet_info[i].mass;
        float3 k3_v = make_float3(0.0f, 0.0f, 0.0f);
        for (size_t j = 0; j < star_num; j++) {
            k3_v = k3_v + calGravity(tmp3.pos, star_info[j].pos, tmp3.mass, star_info[j].mass);
        }
        k3_v = k3_v / tmp3.mass;
        float3 k3_p = v0 + 0.5f * dt * k2_v;

        StarInfo tmp4;
        tmp4.pos = p0 + dt * k3_p;
        tmp4.vel = v0 + dt * k3_v;
        tmp4.mass = planet_info[i].mass;
        float3 k4_v = make_float3(0.0f, 0.0f, 0.0f);
        for (size_t j = 0; j < star_num; j++) {
            k4_v = k4_v + calGravity(tmp4.pos, star_info[j].pos, tmp4.mass, star_info[j].mass);
        }
        k4_v = k4_v / tmp4.mass;
        float3 k4_p = v0 + dt * k3_v;

        planet_info[i].pos = planet_info[i].pos + (dt / 6.0f) * (k1_p + 2.0f * k2_p + 2.0f * k3_p + k4_p);
        planet_info[i].vel = planet_info[i].vel + (dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
    }
    
}


void UpdateParticleStatus(
    Particle* p_read,
    Particle* p_write,
    CenterStatus* connect_status,
    const int* connect_id_map,
    StarInfo* star_info,
    const unsigned int star_num,
    StarInfo* planet_info,
    const unsigned int ParticleNum,
    const unsigned int blockSize,
    const float delta_time)
{
    unsigned int blockNum = (ParticleNum + blockSize - 1) / blockSize;
    unsigned int tileSize = blockSize * 1;
    unsigned int shared_mem = sizeof(float4) * tileSize;
    /*UpdateParticleStatusDevice<<<blockNum, blockSize, shared_mem>>>(p_read, p_write, connect_status, connect_id_map,
        star_info, star_num, planet_info, ParticleNum, delta_time);*/
    /*UpdateParticleStatusDevice << <blockNum, blockSize, shared_mem >> > (p_read, p_write, connect_status,
        connect_id_map, star_info, star_num, planet_info, ParticleNum, delta_time);*/
    UpdateParticleStatusDeviceWithRotation << <blockNum, blockSize, shared_mem >> > (p_read, p_write,
        connect_status, connect_id_map, star_info, star_num, planet_info, ParticleNum, tileSize, delta_time);
    /*UpdateParticleStatusDeviceSpring << <blockNum, blockSize, shared_mem >> > (p_read, p_write,
        connect_status, connect_id_map, star_info, star_num, planet_info, ParticleNum, delta_time);*/
    cudaDeviceSynchronize();
}


void TidalForceV2(
    Particle* p_read,
    Particle* p_write,
    StarInfo** star_info,
    StarInfo** planet_info,
    CenterStatus* connect_planet_device,
    int* connect_id_map_device,
    const unsigned int star_num,
    const unsigned int planet_num,
    const unsigned int ParticleNum,
    const float delta_time,
    const unsigned int blockSize)
{
    //第一步，更新质心坐标，质量
    std::vector<CenterStatus> connect_planets;
    std::vector<int> connect_id_map;
    UpdateCentriod(p_read, planet_info[HOST_PTR], planet_num, blockSize, connect_planets, connect_id_map);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(connect_planet_device, connect_planets.data(), sizeof(CenterStatus) * connect_planets.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(connect_id_map_device, connect_id_map.data(), sizeof(int) * planet_num,
        cudaMemcpyHostToDevice);

    //第二步，更新质心状态，包括位置，速度
    // 只用欧拉法更新，误差太大
    //UpdateBlockStatus(p_read, ParticleNum, connect_planets, star_info[HOST_PTR], star_num, blockSize);
    ////更新planet_info
    //for (unsigned int i = 0; i < connect_planets.size(); i++) {
    //    unsigned int _id = connect_planets[i].id;
    //    //误差太大
    //    planet_info[HOST_PTR][_id].pos = planet_info[HOST_PTR][_id].pos + delta_time * planet_info[HOST_PTR][_id].vel;
    //    planet_info[HOST_PTR][_id].vel = planet_info[HOST_PTR][_id].vel + delta_time * connect_planets[i].acceleration;
    //    planet_info[HOST_PTR][_id].mass = connect_planets[i].mass;
    //}
    //RK4方法更新
    std::vector<CenterStatus> init_connect_planets = connect_planets;
    std::vector<float3> v0(connect_planets.size());
    for (size_t i = 0; i < connect_planets.size(); i++){
        v0[i] = planet_info[HOST_PTR][connect_planets[i].id].vel;
    }
    std::vector<float3> init_v0 = v0;
    UpdateBlockStatusRK4(p_read, ParticleNum, connect_planets, v0, star_info[HOST_PTR],
        star_num, delta_time, blockSize);
    for (unsigned int i = 0; i < connect_planets.size(); i++) {
        unsigned int _id = connect_planets[i].id;
        float3 dp = connect_planets[i].pos - init_connect_planets[i].pos;
        float3 dv = v0[i] - init_v0[i];
        planet_info[HOST_PTR][_id].pos = planet_info[HOST_PTR][_id].pos + dp;
        planet_info[HOST_PTR][_id].vel = planet_info[HOST_PTR][_id].vel + dv;
        planet_info[HOST_PTR][_id].mass = connect_planets[i].mass;
    }
    cudaMemcpy(planet_info[DEVICE_PTR], planet_info[HOST_PTR], sizeof(StarInfo) * planet_num,
        cudaMemcpyHostToDevice);

    auto test_v1 = planet_info[HOST_PTR][0];
    auto test_v2 = planet_info[HOST_PTR][1];
    std::cout << test_v1.pos.x << std::endl;
    std::cout << test_v2.pos.x << std::endl;
    //第三步，更新所有质点的运动状态
    UpdateParticleStatus(p_read, p_write, connect_planet_device, connect_id_map_device, star_info[DEVICE_PTR],
        star_num, planet_info[DEVICE_PTR], ParticleNum, blockSize, delta_time);
    //cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}


void TidalForceV2Spring(
    Particle* p_read,
    Particle* p_write,
    StarInfo** star_info,
    StarInfo** planet_info,
    CenterStatus* connect_planet_device,
    int* connect_id_map_device,
    const unsigned int star_num,
    const unsigned int planet_num,
    const unsigned int ParticleNum,
    const float delta_time,
    const unsigned int blockSize)
{
    //第一步，更新质心坐标，质量
    std::vector<CenterStatus> connect_planets;
    std::vector<int> connect_id_map;
    connect_id_map.resize(planet_num);
    for (size_t i = 0; i < planet_num; i++){
        connect_planets.push_back(CenterStatus{ planet_info[HOST_PTR][i].pos,
            {0.0, 0.0, 0.0}, planet_info[HOST_PTR][i].mass, planet_info[HOST_PTR][i].id });
        connect_id_map[planet_info[HOST_PTR][i].id] = i;
        connect_planets[i].acceleration = { 0.0f, 0.0f, 0.0f };
        for (size_t j = 0; j < star_num; j++) {
            connect_planets[i].acceleration = connect_planets[i].acceleration
                + calGravity(connect_planets[i].pos, star_info[j]->pos,
                    connect_planets[i].mass, star_info[j]->mass) / connect_planets[i].mass;
        }
    }

    auto test1 = planet_info[HOST_PTR][0];
    auto test2 = planet_info[HOST_PTR][1];
    
    cudaMemcpy(connect_planet_device, connect_planets.data(), sizeof(CenterStatus) * connect_planets.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(connect_id_map_device, connect_id_map.data(), sizeof(int) * planet_num,
        cudaMemcpyHostToDevice);

    //RK4方法更新
    UpdateBlockStatusRK4(planet_info[HOST_PTR], planet_num, star_info[HOST_PTR], star_num, delta_time);

    cudaMemcpy(planet_info[DEVICE_PTR], planet_info[HOST_PTR], sizeof(StarInfo) * planet_num,
        cudaMemcpyHostToDevice);

    auto test_v1 = planet_info[HOST_PTR][0];
    auto test_v2 = planet_info[HOST_PTR][1];
    std::cout << test_v1.pos.x << std::endl;
    std::cout << test_v2.pos.x << std::endl;
    //第三步，更新所有质点的运动状态
    UpdateParticleStatus(p_read, p_write, connect_planet_device, connect_id_map_device, star_info[DEVICE_PTR],
        star_num, planet_info[DEVICE_PTR], ParticleNum, blockSize, delta_time);
    //cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}


void TidalForceV2Host(Particle **particle,
    cudaGraphicsResource **cuda_vbo_resource,
    volatile unsigned int currentRead,
    StarInfo** star_info,
    StarInfo** planet_info,
    CenterStatus* connect_status_device,
    int* connect_id_map,
    const unsigned int star_num,
    const unsigned int planet_num,
    const unsigned int ParticleNum,
    const float delta_time,
    const int block_size)
{
    static int time = 0;
    std::cout << "CUDA kernel end:\t" << time++ << "\n";
    //cudaGraphicsResourceSetMapFlags(cuda_vbo_resource, cudaGraphicsMapFlagNone);
    cudaGraphicsResourceSetMapFlags(cuda_vbo_resource[currentRead],
        cudaGraphicsMapFlagsReadOnly);
    cudaGraphicsResourceSetMapFlags(cuda_vbo_resource[1 - currentRead],
        cudaGraphicsMapFlagsWriteDiscard);
    cudaError_t err = cudaGraphicsMapResources(2, cuda_vbo_resource, 0);
    if (err != cudaSuccess) {
        printf("cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);  // 防止后续错误
    }
    size_t bytes;
    err = cudaGraphicsResourceGetMappedPointer((void **)&particle[currentRead], &bytes,
        cuda_vbo_resource[currentRead]);
    err = cudaGraphicsResourceGetMappedPointer((void**)&particle[1 - currentRead], &bytes,
        cuda_vbo_resource[1 - currentRead]);
    if(err != cudaSuccess){
        printf("cudaGraphicsResourceGetMappedPointer failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int num_blocks = (ParticleNum + block_size - 1) / block_size;
    int shared_mem = sizeof(float4) * block_size;
    TidalForceV2(particle[currentRead], particle[1 - currentRead], star_info, planet_info, connect_status_device,
        connect_id_map, star_num, planet_num, ParticleNum, delta_time, block_size);
    /*TidalForceV2Spring(particle[currentRead], particle[1 - currentRead], star_info, planet_info, connect_status_device,
        connect_id_map, star_num, planet_num, ParticleNum, delta_time, block_size);*/
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaGraphicsUnmapResources(2, cuda_vbo_resource, 0);
    if (err != cudaSuccess) {
        printf("cudaGraphicsUnmapResources failed: %s\n", cudaGetErrorString(err));
    }
    //particle = nullptr;
}

