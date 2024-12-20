/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


//-------------------------------------------------------------------------------------------------
// Various common functions.


#ifndef RAYCOMMON_GLSL
#define RAYCOMMON_GLSL

const float InvalidPdf = -1.0;
const uint invalidInCellIndex = 0xFFFFFFFF;

//-----------------------------------------------------------------------
// Debugging
//-----------------------------------------------------------------------
vec3 IntegerToColor(uint val)
{
  const vec3 freq = vec3(1.33333f, 2.33333f, 3.33333f);
  return vec3(sin(freq * val) * .5 + .5);
}

// utility for temperature
float fade(float low, float high, float value)
{
  float mid   = (low + high) * 0.5;
  float range = (high - low) * 0.5;
  float x     = 1.0 - clamp(abs(mid - value) / range, 0.0, 1.0);
  return smoothstep(0.0, 1.0, x);
}

// Return a cold-hot color based on intensity [0-1]
vec3 temperature(float intensity)
{
  const vec3 blue   = vec3(0.0, 0.0, 1.0);
  const vec3 cyan   = vec3(0.0, 1.0, 1.0);
  const vec3 green  = vec3(0.0, 1.0, 0.0);
  const vec3 yellow = vec3(1.0, 1.0, 0.0);
  const vec3 red    = vec3(1.0, 0.0, 0.0);

  vec3 color = (fade(-0.25, 0.25, intensity) * blue    //
                + fade(0.0, 0.5, intensity) * cyan     //
                + fade(0.25, 0.75, intensity) * green  //
                + fade(0.5, 1.0, intensity) * yellow   //
                + smoothstep(0.75, 1.0, intensity) * red);
  return color;
}

//-----------------------------------------------------------------------
// Return the UV in a lat-long HDR map
//-----------------------------------------------------------------------
vec2 GetSphericalUv(vec3 v)
{
  float gamma = asin(-v.y);
  float theta = atan(v.z, v.x);

  vec2 uv = vec2(theta * M_1_OVER_PI * 0.5, gamma * M_1_OVER_PI) + 0.5;
  return uv;
}


//-----------------------------------------------------------------------
// Return the tangent and binormal from the incoming normal
//-----------------------------------------------------------------------
void CreateCoordinateSystem(in vec3 N, out vec3 Nt, out vec3 Nb)
{
  // http://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors.html#CoordinateSystemfromaVector
  //if(abs(N.x) > abs(N.y))
  //  Nt = vec3(-N.z, 0, N.x) / sqrt(N.x * N.x + N.z * N.z);
  //else
  //  Nt = vec3(0, N.z, -N.y) / sqrt(N.y * N.y + N.z * N.z);
  //Nb = cross(N, Nt);

  Nt = normalize(((abs(N.z) > 0.99999f) ? vec3(-N.x * N.y, 1.0f - N.y * N.y, -N.y * N.z) :
                                          vec3(-N.x * N.z, -N.y * N.z, 1.0f - N.z * N.z)));
  Nb = cross(Nt, N);
}


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-----------------------------------------------------------------------
vec3 OffsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}

// compact hdr color to 32bit
// TODO: change tone mapping
uint packYCbCr(in vec3 c) {
    // c = pow(c, vec3(0.5, 0.5, 0.5));
    c = c / (1.0 + c);
    float y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    float cb = -0.16873589 * c.r - 0.33126411 * c.g + 0.5 * c.b + 0.5;
    float cr = 0.5 * c.r - 0.41868759 * c.g - 0.08131241 * c.b + 0.5;
    uint outVal = uint(y * 65535.0) << 16;
    outVal += uint(cb * 255.0) << 8;
    outVal += uint(cr * 255.0);
    return outVal;
}
vec3 unpackYCbCr(in uint c) {
    float y = (c >> 16) / 65535.0;
    float cb = ((c << 16) >> 24) / 255.0 - 0.5;
    float cr = ((c << 24) >> 24) / 255.0 - 0.5;

    float b = cb * 1.772 + y;
    float r = y + 1.402 * cr;
    float g = (y - 0.299 * r - 0.114 * b) / 0.587;
    vec3 rgb = vec3(r, g, b);
    // return pow(rgb / (1.0 - rgb), vec3(2,2,2));
    return rgb / (1.0 - rgb);
}

uint hash8bit(uint a) {
    return (a ^ (a >> 8)) << 24;
}

float getDepth(float z) { // untested
    // return 2.0 * CAMERA_NEAR * CAMERA_FAR / (CAMERA_FAR + CAMERA_NEAR - z_n * (CAMERA_FAR - CAMERA_NEAR));
    float A = CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);
    float B = CAMERA_NEAR * CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);
    return (z - B) / A;
}
float getZ(float depth) { // untested
    return (CAMERA_FAR + CAMERA_NEAR) / (CAMERA_NEAR * CAMERA_FAR) * 0.5 + 0.5 - (CAMERA_FAR * CAMERA_NEAR / depth);
}

uint packTangent(vec3 n, vec3 t) {
    vec3 T, B;
    CreateCoordinateSystem(n, T, B);
    float theta = acos(dot(t, T)) / M_PI;
    float phi = acos(dot(t, B));
    if (phi > M_PI_2) theta = -theta;

    return uint((theta + 1.0) * 32767.499);
}
vec3 unpackTangent(vec3 n, uint val) {
    vec3 T, B;
    CreateCoordinateSystem(n, T, B);
    float theta = (float(val & 0xFFFF) / 32767.499 - 1.0) * M_PI;
    return normalize(cos(theta) * T + sin(theta) * B);
}

vec2 toConcentricDisk(vec2 r) {
    float rx = sqrt(r.x);
    float theta = r.y * 2.0 * M_PI;
    return vec2(cos(theta), sin(theta)) * rx;
}

float powerHeuristic(float f, float g) {
    float f2 = f * f;
    return f2 / (f2 + g * g);
}

bool hasNan(vec3 v) {
    return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

bool inBound(ivec2 p, ivec2 pMin, ivec2 pMax) {
    return p.x >= pMin.x && p.x < pMax.x && p.y >= pMin.y && p.y < pMax.y;
}

bool inBound(ivec2 p, ivec2 bound) {
    return inBound(p, ivec2(0, 0), bound);
}

vec3 HDRToLDR(vec3 color) {
    return color / (color + 1.0);
}

vec3 LDRToHDR(vec3 color) {
    return color / (1.01 - color);
}

vec3 colorWheel(float x)
{
    const float Div = 1.0 / 4.0;
    if (x < Div)
        return vec3(0.0, x / Div, 1.0);
    else if (x < Div * 2)
        return vec3(0.0, 1.0, 2.0 - x / Div);
    else if (x < Div * 3)
        return vec3(x / Div - 2.0, 1.0, 0.0);
    else
        return vec3(1.0, 4.0 - x / Div, 0.0);
}

float Luminance(vec3 rgb)
{
    return 0.3126f * rgb.x + 0.3152f * rgb.y + 0.3722f * rgb.z;
}

//float Luminance(vec3 rgb)
//{
//    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
//}

//
//--------------------------------------------------------------------------------------------------
//
//
vec3 randomColor(vec3 _in) {
    // Generate a pseudo-random color based on the input vector
    float r = fract(sin(dot(_in, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
    float g = fract(sin(dot(_in, vec3(93.989, 67.345, 24.123))) * 43758.5453);
    float b = fract(sin(dot(_in, vec3(29.358, 48.321, 99.123))) * 43758.5453);

    return vec3(r, g, b);
}

#endif  // RAYCOMMON_GLSL
