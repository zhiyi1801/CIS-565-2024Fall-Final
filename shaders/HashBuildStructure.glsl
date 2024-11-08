struct HashAppendData
{
    uint isValid;
    uint reservoirIdx;
    uint cellIdx;
    uint inCellIdx;
};

uint pcg32(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint BinaryNorm(float3 norm)
{
    uint a = norm.x > 0.f ? 1 : 0;
    uint b = norm.y > 0.f ? 1 : 0;
    uint c = norm.z > 0.f ? 1 : 0;
    return a * 100 + b * 10 + c;
    //uint3 biNorm = floor((norm + 1.f) * 1.5f);
    //return biNorm.x << 4 | biNorm.y << 2 | biNorm.z;
}
