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


/*
  Various structure used by CPP and GLSL 
*/


#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE


#ifdef __cplusplus
#include <stdint.h>
// GLSL Type
using ivec2 = glm::ivec2;
using vec2  = glm::vec2;
using vec3  = glm::vec3;
using vec4  = glm::vec4;
using mat4  = glm::mat4;
using uint  = unsigned int;
#endif

// clang-format off
#ifdef __cplusplus  // Descriptor binding helper for C++ and GLSL
#define START_ENUM(a)                                                                                               \
  enum a                                                                                                               \
  {
#define END_ENUM() }
#else
#define START_ENUM(a) const uint
#define END_ENUM()
#endif

// Sets
START_ENUM(SetBindings)
  S_ACCEL = 0,  // Acceleration structure
  S_OUT   = 1,  // Offscreen output image
  S_SCENE = 2,  // Scene data
  S_ENV   = 3,  // Environment / Sun & Sky
  S_RESTIR = 4,  // ReSTIR
  S_WF    = 5   // Wavefront extra data
END_ENUM();

// Acceleration Structure - Set 0
START_ENUM(AccelBindings)
  eTlas = 0 
END_ENUM();

// Output image - Set 1
START_ENUM(OutputBindings)
  eSampler = 0,  // As sampler
  eStore   = 1   // As storage
END_ENUM();

// Scene Data - Set 2
START_ENUM(SceneBindings)
	eCamera = 0,
	eMaterials = 1,
	eInstData = 2,
	ePuncLights = 3,
	eTrigLights = 4,
	eLightBufInfo = 5,
	eTextures = 6  // must be last elem               
END_ENUM();

// Environment - Set 3
START_ENUM(EnvBindings)
  eSunSky     = 0, 
  eHdr        = 1, 
  eImpSamples = 2 
END_ENUM();

// ReSTIR - Set 4
START_ENUM(ReSTIRBindings)
eLastGbuffer = 0,
eCurrentGbuffer = 1,
eInitialReservoirs = 2,
eReservoirs = 3,
eAppend = 4,
eFinal = 5,
eCell = 6,
eIndex = 7,
eCheckSum = 8,
eCellCounter = 9,
eTestImage = 16
END_ENUM();

START_ENUM(DebugMode)
  eNoDebug   = 0,   //
  eBaseColor = 1,   //
  eNormal    = 2,   //
  eMetallic  = 3,   //
  eEmissive  = 4,   //
  eAlpha     = 5,   //
  eRoughness = 6,   //
  eTexcoord  = 7,   //
  eTangent   = 8,   //
  eRadiance  = 9,   //
  eWeight    = 10,  //
  eRayDir    = 11,  //
  eHeatmap   = 12,   //
  eDepth	 = 13
END_ENUM();
// clang-format on


// Camera of the scene
#define CAMERA_NEAR 0.001f
#define CAMERA_FAR 1000.0f
struct SceneCamera
{
  mat4  viewInverse;
  mat4  projInverse;
  float focalDist;
  float aperture;
  // Extra
  int nbLights;
};

struct VertexAttributes
{
  vec3 position;
  uint normal;    // compressed using oct
  vec2 texcoord;  // Tangent handiness, stored in LSB of .y
  uint tangent;   // compressed using oct
  uint color;     // RGBA
};


// GLTF material
#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1
#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2
#define MAX_IOR_MINUS_ONE 3.f
struct GltfShadeMaterial
{
  // 0
  vec4 pbrBaseColorFactor;
  // 4
  int   pbrBaseColorTexture;
  float pbrMetallicFactor;
  float pbrRoughnessFactor;
  int   pbrMetallicRoughnessTexture;
  // 8
  int emissiveTexture;
  int _pad0;
  // 10
  vec3 emissiveFactor;
  int  alphaMode;
  // 14
  float alphaCutoff;
  int   doubleSided;
  int   normalTexture;
  float normalTextureScale;
  // 18
  mat4 uvTransform;
  // 22
  int unlit;

  float transmissionFactor;
  int   transmissionTexture;

  float ior;
  // 26
  vec3  anisotropyDirection;
  float anisotropy;
  // 30
  vec3  attenuationColor;
  float thicknessFactor;  // 34
  int   thicknessTexture;
  float attenuationDistance;
  // --
  float clearcoatFactor;
  float clearcoatRoughness;
  // 38
  int  clearcoatTexture;
  int  clearcoatRoughnessTexture;
  uint sheen;
  int  _pad1;
  // 42
};


// Use with PushConstant
struct RtxState
{
  int   frame;                  // Current frame, start at 0
  int   maxDepth;               // How deep the path is
  int   maxSamples;             // How many samples to do per render
  float fireflyClampThreshold;  // to cut fireflies

  float hdrMultiplier;          // To brightening the scene
  int   debugging_mode;         // See DebugMode
  int   pbrMode;                // 0-Disney, 1-Gltf
  int   _pad0;                  // vec2 need alignment

  ivec2 size;                   // rendering size
  int   minHeatmap;             // Debug mode - heat map
  int   maxHeatmap;

  // Light Info
  float envMapLuminIntegInv;
  float lightLuminIntegInv;
  float environmentProb; // Used in direct light importance sampling
  int MIS;			   // 0 - off, 1 - on
};

// Structure used for retrieving the primitive information in the closest hit
// using gl_InstanceCustomIndexNV
struct InstanceData
{
  uint64_t vertexAddress;
  uint64_t indexAddress;
  int      materialIndex;
};


// KHR_lights_punctual extension.
// see https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual

const int LightType_Directional = 0;
const int LightType_Point       = 1;
const int LightType_Spot        = 2;

struct Light
{
  vec3  direction;
  float range;

  vec3  color;
  float intensity;

  vec3  position;
  float innerConeCos;

  float outerConeCos;
  int   type;

  vec2 padding;
};

// Environment acceleration structure - computed in hdr_sampling
struct EnvAccel
{
  uint  alias;
  float q;
  float pdf;
  float aliasPdf;
};

// Tonemapper used in post.frag
struct Tonemapper
{
  float brightness;
  float contrast;
  float saturation;
  float vignette;
  float avgLum;
  float zoom;
  vec2  renderingRatio;
  int   autoExposure;
  float Ywhite;  // Burning white
  float key;     // Log-average luminance
  int   dither;
};


struct SunAndSky
{
  vec3  rgb_unit_conversion;
  float multiplier;

  float haze;
  float redblueshift;
  float saturation;
  float horizon_height;

  vec3  ground_color;
  float horizon_blur;

  vec3  night_color;
  float sun_disk_intensity;

  vec3  sun_direction;
  float sun_disk_scale;

  float sun_glow_intensity;
  int   y_is_up;
  int   physically_scaled_sun;
  int   in_use;
};

// ReSTIR Structure
struct Reservoir
{
    vec3 vPos;
	uint M;

    vec3 vNorm;
	float weightF; //the weight use for compute final illuminance W = Weight / (Mount * pdf)

    vec3 sPos;
	uint age; //age the sample  > maxSampleAge will be discard

    vec3 sNorm;
	int pad1;

    vec3 radiance;
	int pad2;
};

struct FinalSample
{
    vec3 dir;
	int pad1;

    vec3 Li;
	int pad2;
};

struct HashAppendData
{
    uint isValid;
    uint reservoirIdx;
    uint cellIdx;
    uint inCellIdx;
};

// Light sampling data
// acceleration structure for importance sampling - pre-computed
// Alias method for importance sampling
struct ImptSampData
{
	int alias;
	float q;
	float pdf;
	float aliasPdf;
};

struct LightBufInfo
{
    uint puncLightSize;
    uint trigLightSize;
    float trigSampProb;
    int pad;
};

struct PuncLight // point, spot, or directional light.
{
	int type;
	vec3 direction;

	float intensity;
	vec3 color;

	vec3 position;
	float range;

	float outerConeCos;
	float innerConeCos;
	vec2 padding;

	ImptSampData impSamp;
};

struct TrigLight
{ // triangles of emissive meshes
	uint matIndex;
	uint transformIndex;
	vec3 v0;
	vec3 v1;
	vec3 v2;
	vec2 uv0;
	vec2 uv1;
	vec2 uv2;
	ImptSampData impSamp;
	vec3 pad;
};

#endif  // COMMON_HOST_DEVICE
