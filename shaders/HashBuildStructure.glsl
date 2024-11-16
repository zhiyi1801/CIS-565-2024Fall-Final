#ifndef HASH_BUILD_STRUCTURE_GLSL
#define HASH_BUILD_STRUCTURE_GLSL

#include "host_device.h"

// second hash index
uint jenkinsHash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Basic hash function
uint pcg32(uint _input)
{
    uint state = _input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// quantize a normal to a uint
uint BinaryNorm(vec3 norm)
{
    uint a = norm.x > 0.f ? 1 : 0;
    uint b = norm.y > 0.f ? 1 : 0;
    uint c = norm.z > 0.f ? 1 : 0;
    return a * 100 + b * 10 + c;
    //uint3 biNorm = floor((norm + 1.f) * 1.5f);
    //return biNorm.x << 4 | biNorm.y << 2 | biNorm.z;
}

// Get the cell size
float CalculateCellSize(vec3 pos, vec3 cameraPos, GIParameter params)
{
    // https://wangningbei.github.io/2023/ReSTIR_files/paper_ReSTIRGI.pdf
    // https://dl.acm.org/doi/10.1145/3478512.3488613

    float cellSizeStep = length(pos - cameraPos) * tan(120 * params.fov * max(1.0 / params.frameDim.y, params.frameDim.y / float((params.frameDim.x * params.frameDim.x))));
    int logStep = int(floor(log2(cellSizeStep / params.minCellSize)));

    return params.minCellSize * max(0.12f, exp2(logStep));
}



#endif // HASH_BUILD_STRUCTURE_GLSL