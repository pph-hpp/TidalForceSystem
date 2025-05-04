#include "ForceCalc.cuh"

__host__ __device__ float calDistance3D(const float3& a, const float3& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

__host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator*(float s, float3 v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__host__ __device__ float3 operator/(float3 s, float v) {
    return make_float3(s.x / v, s.y / v, s.z / v);
}

__device__ bool swallowDetection(const float3* position) {
    return ((calDistance3D(*position, { 0, 0, 0 }) <= BLOCK_HOLE_RADIUS));
}

__device__ void SwallowedByStar(Particle* p, const StarInfo* star_info, unsigned int star_num) {
    for (unsigned int i = 0; i < star_num; i++){
        if (calDistance3D(p->position, star_info[i].pos) <= star_info[i].radius) {
            p->is_swallowed = true;
            p->position = star_info[i].pos;
            return;
        }
    }
}

__host__ __device__ float calLengthOfVec(const float3& a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__ float3 calGravity(const float3& p_target, const float3& p_source, const float mass1, const float mass2) {
    float length = calDistance3D(p_target, p_source);
    float force_scale;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    if (length <= 0.01) {
        //缺少碰撞检测
        return { 0.0f, 0.0f, 0.0f };
        /*force_scale = G * mass1 * mass2 / (8 * PARTICLE_RADIUS * PARTICLE_RADIUS * PARTICLE_RADIUS);
        force.x = force_scale * (p_source.x - p_target.x) / length;
        force.y = force_scale * (p_source.y - p_target.y) / length;
        force.z = force_scale * (p_source.z - p_target.z) / length;*/
    }
    else {
        force_scale = G * mass1 * mass2 / (length * length * length);
        force.x = force_scale * (p_source.x - p_target.x);
        force.y = force_scale * (p_source.y - p_target.y);
        force.z = force_scale * (p_source.z - p_target.z);
    }
    return force;
}


__host__ __device__ float3 calGravityV3(const float3& p_target, const float3& p_source, const float mass1, const float mass2,
    const float3& center_pos, const float radius) 
{
    float length = calDistance3D(p_target, p_source);
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    //使用软化参数防止近距离下数值不稳定
    float e = 0.1;
    float force_scale = G * mass1 * mass2 / pow(sqrt(length * length + e * e), 3);

    force.x = force_scale * (p_source.x - p_target.x);
    force.y = force_scale * (p_source.y - p_target.y);
    force.z = force_scale * (p_source.z - p_target.z);
    if (calDistance3D(center_pos, p_target) <= 1.2 * radius){
        return { 0.0f, 0.0f, 0.0f };
    }
    return force;
}


__host__ __device__ float3 updateVec(const float3* vec,
    const float3* offset,
    const float deltaTime) {
    float3 vector{ vec->x + offset->x * deltaTime,
        vec->y + offset->y * deltaTime,
        vec->z + offset->z * deltaTime };
    return vector;
}

__device__ float3 cross(const float3& r1, const float3& r2) {
    return make_float3(
        r1.y * r2.z - r1.z * r2.y,
        r1.z * r2.x - r1.x * r2.z,
        r1.x * r2.y - r1.y * r2.x
    );
}

__device__ float dot(const float3& r1, const float3& r2) {
    return (r1.x * r2.x + r1.y * r2.y + r1.z * r2.z);
}

__device__ float3 CalSpinCentrifugal(const float3& w, const float3& r) {
    return dot(w, w) * r - dot(w, r) * w;
}

//p:相对球心的相对坐标
//n:
__device__ float3 rotateRodrigues(float3 p, float3 v_spin, float delta_time) {
    float spin_len = calLengthOfVec(v_spin);
    if (spin_len < 1e-8f) return p; // 无旋转，直接返回原始向量
    float theta = spin_len * delta_time;
    float3 n = v_spin / spin_len;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float3 term1 = cos_theta * p;
    float3 term2 = sin_theta * cross(n, p);
    float3 term3 = dot(n, p) * (1.0f - cos_theta) * n;

    return term1 + term2 + term3;
}

__device__ int flatten2D(int i, int j, int num) {
    return i * num + j;
}

__device__ void CalTotalForce(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const float particle_mass,
    float3* totalForce)
{
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    extern __shared__ float3 shared_positions[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float3 gravity;
    #pragma unroll
    for (int i = 0; i < ParticleNum; i += tileSize) {
        if (i + tid < ParticleNum) {
            shared_positions[tid] = particles[i + tid].position;
        }
        __syncthreads();

        int size = (ParticleNum - i) < tileSize ? (ParticleNum - i) : tileSize;
#pragma unroll
        for (int p_idx = 0; p_idx < size; p_idx++) {
            if (i + p_idx == p->index) {
                continue;
            }
            //every particle is same
            gravity = calGravity(p->position, shared_positions[p_idx], particle_mass, particle_mass);
            total_force = updateVec(&total_force, &gravity);
        }
    }
    gravity = calGravity(p->position, STAR_POSITION, particle_mass, STAR_MASS);
    total_force = updateVec(&total_force, &gravity);
    (*totalForce) = total_force;
}



__device__ void CalTotalForceV2(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const StarInfo* star_info,
    unsigned int star_num,
    float3* totalForce)
{
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    extern __shared__ float4 shared_pos_and_mass[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = blockDim.x;
    float3 pos = p->position;
    float3 gravity;
    //#pragma unroll
    for (int i = 0; i < ParticleNum; i += tileSize) {
        for (size_t j = 0; j < tileSize / n; j++){
            shared_pos_and_mass[tid + j * n] = (i + tid + j * n < ParticleNum) ? make_float4(
                particles[i + tid + j * n].position.x,
                particles[i + tid + j * n].position.y,
                particles[i + tid + j * n].position.z,
                particles[i + tid + j * n].mass) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        int size = (ParticleNum - i) < tileSize ? (ParticleNum - i) : tileSize;
        //#pragma unroll
        for (int p_idx = 0; p_idx < size; p_idx++) {
            if (shared_pos_and_mass[p_idx].w > 1e-7 && i + p_idx != p->index) {
                gravity = calGravity(pos, { shared_pos_and_mass[p_idx].x, shared_pos_and_mass[p_idx].y, shared_pos_and_mass[p_idx].z },
                    p->mass, shared_pos_and_mass[p_idx].w);
                total_force = updateVec(&total_force, &gravity);
            }
        }
    }
    for (int i = 0; i < star_num; i++) {
        gravity = calGravity(pos, star_info[i].pos, p->mass, star_info[i].mass);
        total_force = updateVec(&total_force, &gravity);
    }
    (*totalForce) = total_force;
}



__device__ void CalTotalForceV3(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const StarInfo* star_info,
    const StarInfo* planet_info,
    unsigned int star_num,
    float3* totalForce)
{
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    extern __shared__ float4 shared_pos_and_mass[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float3 gravity;
    //#pragma unroll
    for (int i = 0; i < ParticleNum; i += tileSize) {
        shared_pos_and_mass[tid].x = (i + tid < ParticleNum) ? particles[i + tid].position.x : 0.0f;
        shared_pos_and_mass[tid].y = (i + tid < ParticleNum) ? particles[i + tid].position.y : 0.0f;
        shared_pos_and_mass[tid].z = (i + tid < ParticleNum) ? particles[i + tid].position.z : 0.0f;
        shared_pos_and_mass[tid].w = (i + tid < ParticleNum) ? particles[i + tid].mass : 0.0f;
        __syncthreads();

        int size = (ParticleNum - i) < tileSize ? (ParticleNum - i) : tileSize;
        //#pragma unroll
        for (int p_idx = 0; p_idx < size; p_idx++) {
            if (shared_pos_and_mass[p_idx].w > 1e-7 && i + p_idx != p->index) {
                gravity = calGravityV3(p->position, { shared_pos_and_mass[p_idx].x, shared_pos_and_mass[p_idx].y, shared_pos_and_mass[p_idx].z },
                    p->mass, shared_pos_and_mass[p_idx].w, planet_info->pos, planet_info->radius);
                total_force = updateVec(&total_force, &gravity);
            }
        }
    }
    for (int i = 0; i < star_num; i++) {
        gravity = calGravityV3(p->position, star_info[i].pos, p->mass, star_info[i].mass,
            planet_info->pos, planet_info->radius);
        total_force = updateVec(&total_force, &gravity);
    }
    (*totalForce) = total_force;
    //(*totalForce) = { 0.0f, 0.0f, 0.0f };
}


__device__ void CalTotalForce_v2(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const float particle_mass,
    float3* totalForce)
{
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    extern __shared__ float shared_mem[];

    float* shared_pos_x = shared_mem;
    float* shared_pos_y = shared_mem + blockDim.x + 1;
    float* shared_pos_z = shared_pos_y + blockDim.x + 1;
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float3 gravity;
    //#pragma unroll
    for (int i = 0; i < ParticleNum; i += tileSize) {
        if (i + tid < ParticleNum) {
            float3 pos = particles[i + tid].position;
            shared_pos_x[tid] = pos.x;
            shared_pos_y[tid] = pos.y;
            shared_pos_z[tid] = pos.z;
        }
        __syncthreads();

        int size = (ParticleNum - i) < tileSize ? (ParticleNum - i) : tileSize;
#pragma unroll
        for (int p_idx = 0; p_idx < size; p_idx++) {
            if (i + p_idx == p->index) {
                continue;
            }
            //every particle is same
            gravity = calGravity(p->position, { shared_pos_x[p_idx], shared_pos_y[p_idx], shared_pos_z[p_idx] },
                particle_mass, particle_mass);
            total_force = updateVec(&total_force, &gravity);
        }
    }
    gravity = calGravity(p->position, STAR_POSITION, particle_mass, STAR_MASS);
    total_force = updateVec(&total_force, &gravity);
    (*totalForce) = total_force;
}


__device__ void CalTotalForce_v3(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const float particle_mass,
    float3* totalForce)
{
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    extern __shared__ float4 shared_m[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float3 gravity;
    //#pragma unroll
    for (int i = 0; i < ParticleNum; i += tileSize) {
        if (i + tid < ParticleNum) {
            float3 pos = particles[i + tid].position;
            shared_m[tid] = { pos.x, pos.y, pos.z, 0.0f };
        }
        __syncthreads();

        int size = (ParticleNum - i) < tileSize ? (ParticleNum - i) : tileSize;
#pragma unroll
        for (int p_idx = 0; p_idx < size; p_idx++) {
            if (i + p_idx == p->index) {
                continue;
            }
            //every particle is same
            gravity = calGravity(p->position, { shared_m[p_idx].x, shared_m[p_idx].y, shared_m[p_idx].z },
                particle_mass, particle_mass);
            total_force = updateVec(&total_force, &gravity);
        }
    }
    gravity = calGravity(p->position, STAR_POSITION, particle_mass, STAR_MASS);
    total_force = updateVec(&total_force, &gravity);
    (*totalForce) = total_force;
}


__device__ void updateParticleRK4(Particle& p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const float particle_mass,
    float dt)
{
    if (!p.is_swallowed) {
        float3 p0 = p.position;
        float3 v0 = p.velocity;
        float3 a0;
        //CalTotalForce(&p, particles, tileSize, ParticleNum, particle_mass, &a0);
        CalTotalForce_v2(&p, particles, tileSize, ParticleNum, particle_mass, &a0);
        float3 k1_v = a0;
        float3 k1_p = v0;

        Particle tmp2 = p;
        tmp2.position = p0 + 0.5f * dt * k1_p;
        tmp2.velocity = v0 + 0.5f * dt * k1_v;
        float3 k2_v;
        //CalTotalForce(&tmp2, particles, tileSize, ParticleNum, particle_mass, &k2_v);
        CalTotalForce_v2(&tmp2, particles, tileSize, ParticleNum, particle_mass, &k2_v);
        float3 k2_p = v0 + 0.5f * dt * k1_v;

        Particle tmp3;
        tmp3.position = p0 + 0.5f * dt * k2_p;
        tmp3.velocity = v0 + 0.5f * dt * k2_v;
        float3 k3_v;
        //CalTotalForce(&tmp3, particles, tileSize, ParticleNum, particle_mass, &k3_v);
        CalTotalForce_v2(&tmp3, particles, tileSize, ParticleNum, particle_mass, &k3_v);
        float3 k3_p = v0 + 0.5f * dt * k2_v;

        Particle tmp4;
        tmp4.position = p0 + dt * k3_p;
        tmp4.velocity = v0 + dt * k3_v;
        float3 k4_v;
        //CalTotalForce(&tmp4, particles, tileSize, ParticleNum, particle_mass, &k4_v);
        CalTotalForce_v2(&tmp4, particles, tileSize, ParticleNum, particle_mass, &k4_v);
        float3 k4_p = v0 + dt * k3_v;

        if (!p.is_connect){
            p.position = p.position + (dt / 6.0f) * (k1_p + 2.0f * k2_p + 2.0f * k3_p + k4_p);
            p.velocity = p.velocity + (dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
        }
        
    }
}


__device__ void updateParticleRK4V2(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const StarInfo* star_info,
    const int star_num,
    float dt)
{
    if (!p->is_swallowed) {
        float3 p0 = p->position;
        float3 v0 = p->velocity;
        float3 a0;
        CalTotalForceV2(p, particles, tileSize, ParticleNum, star_info, star_num, &a0);
        a0 = a0 / p->mass;

        float3 k1_v = a0;
        float3 k1_p = v0;

        Particle tmp2 = *p;
        float3 k2_v;
        CalTotalForceV2(&tmp2, particles, tileSize, ParticleNum, star_info, star_num, &k2_v);
        k2_v = k2_v / p->mass;
        float3 k2_p = v0 + 0.5f * dt * k1_v;

        Particle tmp3;
        tmp3.position = p0 + 0.5f * dt * k2_p;
        tmp3.velocity = v0 + 0.5f * dt * k2_v;
        tmp3.mass = p->mass;
        float3 k3_v;
        CalTotalForceV2(&tmp3, particles, tileSize, ParticleNum, star_info, star_num, &k3_v);
        k3_v = k3_v / p->mass;
        float3 k3_p = v0 + 0.5f * dt * k2_v;

        Particle tmp4;
        tmp4.position = p0 + dt * k3_p;
        tmp4.velocity = v0 + dt * k3_v;
        tmp4.mass = p->mass;
        float3 k4_v;
        CalTotalForceV2(&tmp4, particles, tileSize, ParticleNum, star_info, star_num, &k4_v);
        k4_v = k4_v / p->mass;
        float3 k4_p = v0 + dt * k3_v;

        if (!p->is_connect) {
            p->position = p->position + (dt / 6.0f) * (k1_p + 2.0f * k2_p + 2.0f * k3_p + k4_p);
            p->velocity = p->velocity + (dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
        }
    }
}


__device__ void updateParticleRK4V3(Particle* p,
    Particle* particles,
    const unsigned int tileSize,
    unsigned int ParticleNum,
    const StarInfo* star_info,
    const StarInfo* planet_info,
    const int star_num,
    float dt)
{
    if (!p->is_swallowed) {
        float3 p0 = p->position;
        float3 v0 = p->velocity;
        float3 a0;
        CalTotalForceV3(p, particles, tileSize, ParticleNum, star_info, planet_info, star_num, &a0);
        a0 = a0 / p->mass;

        float3 k1_v = a0;
        float3 k1_p = v0;

        Particle tmp2 = *p;
        float3 k2_v;
        CalTotalForceV3(&tmp2, particles, tileSize, ParticleNum, star_info, planet_info, star_num, &k2_v);
        k2_v = k2_v / p->mass;
        float3 k2_p = v0 + 0.5f * dt * k1_v;

        Particle tmp3;
        tmp3.position = p0 + 0.5f * dt * k2_p;
        tmp3.velocity = v0 + 0.5f * dt * k2_v;
        tmp3.mass = p->mass;
        float3 k3_v;
        CalTotalForceV3(&tmp3, particles, tileSize, ParticleNum, star_info, planet_info, star_num, &k3_v);
        k3_v = k3_v / p->mass;
        float3 k3_p = v0 + 0.5f * dt * k2_v;

        Particle tmp4;
        tmp4.position = p0 + dt * k3_p;
        tmp4.velocity = v0 + dt * k3_v;
        tmp4.mass = p->mass;
        float3 k4_v;
        CalTotalForceV3(&tmp4, particles, tileSize, ParticleNum, star_info, planet_info, star_num, &k4_v);
        k4_v = k4_v / p->mass;
        float3 k4_p = v0 + dt * k3_v;

        if (!p->is_connect) {
            p->position = p->position + (dt / 6.0f) * (k1_p + 2.0f * k2_p + 2.0f * k3_p + k4_p);
            p->velocity = p->velocity + (dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
        }
    }
}

__device__ void updateCenterRK4(float3* position,
    float3* velocity,
    float dt)
{
    float3 p0 = *position;
    float3 v0 = *velocity;
    float3 a0 = -1 * G * STAR_MASS / (pow(calDistance3D(p0, STAR_POSITION), 3)) * (p0);

    float3 k1_v = a0;
    float3 k1_p = v0;

    float3 p2 = p0 + 0.5f * dt * k1_p;
    float3 k2_v = -1 * G * STAR_MASS / (pow(calDistance3D(p2, STAR_POSITION), 3)) * (p2);
    float3 k2_p = v0 + 0.5f * dt * k1_v;

    float3 p3 = p0 + 0.5f * dt * k2_p;
    float3 k3_v = -1 * G * STAR_MASS / (pow(calDistance3D(p3, STAR_POSITION), 3)) * (p3);
    float3 k3_p = v0 + 0.5f * dt * k2_v;

    float3 p4 = p0 + dt * k3_p;
    float3 k4_v = -1 * G * STAR_MASS / (pow(calDistance3D(p4, STAR_POSITION), 3)) * (p4);
    float3 k4_p = v0 + dt * k3_v;

    (*position) = (*position) + (dt / 6.0f) * (k1_p + 2.0f * k2_p + 2.0f * k3_p + k4_p);
    (*velocity) = (*velocity) + (dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
}


