
#ifndef HASH_BUILD_STRUCTURE_GLSL
#define HASH_BUILD_STRUCTURE_GLSL

#include "host_device.h"
//struct HashAppendData
//{
//    uint isValid;
//    uint reservoirIdx;
//    uint cellIdx;
//    uint inCellIdx;
//};
//
//struct GIParameter
//{
//    uint2 frameDim = { };
//    uint frameCount = 0u;
//    uint instanceID = 0u;
//
//    float3 sceneBBMin = { };
//    float fov = 0.f;
//
//    float4 _pad = { };
//    float minCellSize = 0.0f;   
//};

// TODO: Rewrite RWByteAddressBuffer jenkinsHash pcg32
uint pcg32(uint inp)
{
    uint state = inp * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint BinaryNorm(vec3 norm)
{
    uint a = norm.x > 0.0 ? 1u : 0u;
    uint b = norm.y > 0.0 ? 1u : 0u;
    uint c = norm.z > 0.0 ? 1u : 0u;
    return a * 100u + b * 10u + c;
}

uint jenkinsHash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return uint(a);
}

float CalculateCellSize(vec3 pos, vec3 cameraPos, GIParameter params)
{
    float cellSizeStep = length(pos - cameraPos) * tan(120.0 * params.fov * max(1.0 / float(params.frameDim.y), float(params.frameDim.y) / float(params.frameDim.x * params.frameDim.x)));
    int logStep = int(floor(log2(cellSizeStep / params.minCellSize)));

    return params.minCellSize * max(0.12, exp2(float(logStep)));
}

// RWByteAddressBuffer in slang, but checkSumBuffer in GLSL stucture(should be bind in layout)
int FindOrInsertCell(vec3 pos, vec3 norm, float cellSize, GIParameter params)
{
    uvec3 p = uvec3(
        uint(floor((pos - params.sceneBBMin).x / cellSize)),
        uint(floor((pos - params.sceneBBMin).y / cellSize)),
        uint(floor((pos - params.sceneBBMin).z / cellSize))
    );

    uint normprint = BinaryNorm(norm);

    uint cellIndex = pcg32(normprint + pcg32(uint(cellSize)+pcg32(p.z + pcg32(p.y + pcg32(p.x))))) % 100000u;
    uint checkSum = max(uint(jenkinsHash(normprint + jenkinsHash(uint(cellSize)+jenkinsHash(p.z + jenkinsHash(p.y + jenkinsHash(p.x)))))), 1u);

    for (uint i = 0u; i < 32u; i++)
    {
        uint idx = cellIndex * 32u + i;
        uint expected = 0u;
        uint desired = checkSum;
        uint checkSumPre = atomicCompSwap(checkSumBuffer[idx], expected, desired);
        //uint checkSumPre = atomicCompSwap(checkSumBuffer.checkSumBufferData[idx], expected, desired);

        if (checkSumPre == 0u || checkSumPre == checkSum)
            return int(idx);
    }

    return -1;
}

int FindCell(vec3 pos, vec3 jitteredPos, vec3 norm, float cellSize, GIParameter params)
{
    uvec3 p = uvec3(
        uint(floor((pos - params.sceneBBMin).x / cellSize)),
        uint(floor((pos - params.sceneBBMin).y / cellSize)),
        uint(floor((pos - params.sceneBBMin).z / cellSize))
    );

    uint normprint = BinaryNorm(norm);
    uint cellIndex = pcg32(normprint + pcg32(uint(cellSize)+pcg32(p.z + pcg32(p.y + pcg32(p.x))))) % 100000u;
    uint checkSum = max(uint(jenkinsHash(normprint + jenkinsHash(uint(cellSize)+jenkinsHash(p.z + jenkinsHash(p.y + jenkinsHash(p.x)))))), 1u);

    for (uint i = 0u; i < 32u; i++)
    {
        uint idx = cellIndex * 32u + i;

        // * rewrite the structure
        if (checkSumBuffer[idx] == checkSum)
            return int(idx);
    }

    return -1;
}


int FindCell(vec3 jitteredPos, vec3 norm, float cellSize, GIParameter params)
{

    uvec3 p = uvec3(
        uint(floor((jitteredPos - params.sceneBBMin).x / cellSize)),
        uint(floor((jitteredPos - params.sceneBBMin).y / cellSize)),
        uint(floor((jitteredPos - params.sceneBBMin).z / cellSize))
    );

    uint normprint = BinaryNorm(norm);

    uint cellIndex = pcg32(normprint + pcg32(uint(cellSize)+pcg32(p.z + pcg32(p.y + pcg32(p.x))))) % 100000u;
    uint checkSum = max(uint(jenkinsHash(normprint + jenkinsHash(uint(cellSize)+jenkinsHash(p.z + jenkinsHash(p.y + jenkinsHash(p.x)))))), 1u);

    for (uint i = 0u; i < 32u; i++)
    {
        uint idx = cellIndex * 32u + i;
        // rewrite structure
        if (checkSumBuffer[idx] == checkSum)
            return int(idx);
    }

    return -1;
}

#endif  // HASH_BUILD_STRUCTURE_GLSL
