#include "ParticleStruct.h"


void TidalForceV2Host(Particle** particle,
    cudaGraphicsResource** cuda_vbo_resource,
    volatile unsigned int currentRead,
    StarInfo** star_info,
    StarInfo** planet_info,
    CenterStatus* connect_status_device,
    int* connect_id_map,
    const unsigned int star_num,
    const unsigned int planet_num,
    const unsigned int ParticleNum,
    const float delta_time,
    const int block_size);


void TidalForceV2(
    Particle* p_read,
    Particle* p_write,
    StarInfo** star_info,
    StarInfo** planet_info,
    CenterStatus* connect_planet_device,
    int* connect_id_map,
    const unsigned int star_num,
    const unsigned int planet_num,
    const unsigned int ParticleNum,
    const float delta_time,
    const unsigned int blockSize);