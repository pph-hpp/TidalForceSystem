#pragma once

#define G 6.67430e-11
#define PARTICLE_DENSITY 3.0f
#define BLOCK_HOLE_DENSITY 300000000.0f
#define PI 3.14159265358979323846
#define PARTICLE_RADIUS 1.0f
#define STAR_POSITION make_float3(-1300.0f, 0.0f, 0.0f)
#define BLOCK_HOLE_RADIUS 60.0f
#define STAR_MASS (4.0/3 * PI * BLOCK_HOLE_RADIUS*BLOCK_HOLE_RADIUS*BLOCK_HOLE_RADIUS*BLOCK_HOLE_DENSITY)
#define PLANET_STAR_DISTANCE 300.0f
#define PLANET_RADIUS 20.0f
#define SHEAR_STRENGTH 3000.0f   //剪切强度
//#define TENSILE_STRENGTH 0.1f        //拉伸强度
#define TENSILE_STRENGTH 0.8f        //拉伸强度
#define STAR_POS {0, 0, 0}
#define k_spring_n 50
#define k_spring_t 5000
#define k_spring 0.018

#define HOST_PTR 0
#define DEVICE_PTR 1

struct Particle
{
    float3 position;
    float3 velocity;
    float angle_velocity;
    float total_angle;
    float radius;
    float angle_phi;
    float angle_theta;
    unsigned int is_connect;
    unsigned int is_swallowed;
    unsigned int index;
    unsigned int planet_id;
    float mass;
    //float density;
};

struct StarInfo{
    float3 pos;
    float3 vel;
    float3 v_spin;
    float mass;
    unsigned int offset;
    unsigned int particle_num;
    unsigned int id;
    float radius;
};

struct CenterStatus{
    float3 pos;
    float3 acceleration;
    float mass;
    unsigned int id;
};

//extern float STAR_MASS;

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


