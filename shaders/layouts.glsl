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
// This file holds the layout used by all ray tracing shaders


#ifndef LAYOUTS_GLSL
#define LAYOUTS_GLSL 1


// C++ shared structures and binding
#include "host_device.h"

//----------------------------------------------
// Descriptor Set Layout
//----------------------------------------------


// clang-format off
layout(set = S_ACCEL, binding = eTlas)					uniform accelerationStructureEXT topLevelAS;
//
layout(set = S_OUT,   binding = eStore)					uniform image2D			resultImage;
layout(set = S_OUT, binding = eThisDirectResult)   uniform image2D thisDirectResultImage;
layout(set = S_OUT, binding = eLastDirectResult)   uniform readonly image2D lastDirectResultImage;
//
layout(set = S_SCENE, binding = eInstData, scalar)   buffer _InstanceInfo { InstanceData geoInfo[]; };
layout(set = S_SCENE, binding = eCamera, scalar)   uniform _SceneCamera{ SceneCamera sceneCamera; };
layout(set = S_SCENE, binding = eMaterials, scalar)		buffer _MaterialBuffer { GltfShadeMaterial materials[]; };
layout(set = S_SCENE, binding = ePuncLights, scalar)		buffer _PuncLights { PuncLight puncLights[]; };
layout(set = S_SCENE, binding = eTrigLights, scalar)		buffer _TrigLights { TrigLight trigLights[]; };
layout(set = S_SCENE, binding = eLightBufInfo)		uniform _LightBufInfo{ LightBufInfo lightBufInfo; };
layout(set = S_SCENE, binding = eTextures)   uniform sampler2D		texturesMap[];
//
layout(set = S_ENV, binding = eSunSky,		scalar)		uniform _SSBuffer		{ SunAndSky _sunAndSky; };
layout(set = S_ENV, binding = eHdr)						uniform sampler2D		environmentTexture;
layout(set = S_ENV, binding = eImpSamples,  scalar)		buffer _EnvAccel		{ EnvAccel envSamplingData[]; };

layout(set = S_RESTIR, binding = eLastGbuffer)       uniform readonly uimage2D lastGbuffer;
layout(set = S_RESTIR, binding = eCurrentGbuffer)       uniform uimage2D thisGbuffer;
layout(set = S_RESTIR, binding = eMotionVector)      uniform iimage2D motionVector;

// ReSTIR DI
layout(set = S_RESTIR, binding = ePrevDirectReservoirs, scalar) buffer _PrevDirectResv { DirectReservoir prevDirectResv[]; };
layout(set = S_RESTIR, binding = eCurrentDirectReservoirs, scalar) buffer _CurrentDirectResv { DirectReservoir currentDirectResv[]; };

layout(set = S_RESTIR, binding = eInitialReservoirs, scalar)   buffer _InitialReservoirs { Reservoir initialReserviors[]; };
layout(set = S_RESTIR, binding = eCurrentReservoirs, scalar)   buffer _CurrentReservoirs { Reservoir currentReserviors[]; };
layout(set = S_RESTIR, binding = ePrevReservoirs, scalar)   buffer _PrevReservoirs { Reservoir prevReserviors[]; };
layout(set = S_RESTIR, binding = eAppend, scalar)   buffer _AppendBuffer { HashAppendData appendBuffer[]; };
layout(set = S_RESTIR, binding = eFinal, scalar)   buffer _FinalSamples { FinalSample finalSamples[]; };
layout(set = S_RESTIR, binding = eCell, scalar)   buffer _CellStorageBuffer { uint cellStorageBuffer[]; };
layout(set = S_RESTIR, binding = eIndex, scalar)   buffer _IndexBuffer { uint indexBuffer[]; };
layout(set = S_RESTIR, binding = eCheckSum, scalar)   buffer _CheckSumBuffer { uint checkSumBuffer[]; };
layout(set = S_RESTIR, binding = eCellCounter, scalar)   buffer _CellCounterBuffer { uint cellCounterBuffer[]; };
layout(set = S_RESTIR, binding = eInitialSamples, scalar)   buffer _InitialSampleBuffer { InitialSample initialSampleBuffer[]; };
layout(set = S_RESTIR, binding = eReconnection, scalar)   buffer _ReconnectionDataBuffer { ReconnectionData reconnectionDataBuffer[]; };
layout(set = S_RESTIR, binding = eIndexTemp, scalar)   buffer _IndexTempBuffer { uint indexTempBuffer[]; };

layout(set = S_RESTIR, binding = eDebugUintImage)					uniform uimage2D			debugUintImage;
layout(set = S_RESTIR, binding = eDebugImage)						uniform image2D			debugImage;
layout(set = S_RESTIR, binding = eDebugUintBuffer)					buffer _DebugUintBuffer { uint debugUintBuffer[]; };
layout(set = S_RESTIR, binding = eDebugFloatBuffer)					buffer _DebugFloatBuffer { float debugFloatBuffer[]; };

layout(buffer_reference, scalar) buffer Vertices { VertexAttributes v[]; };
layout(buffer_reference, scalar) buffer Indices	 { uvec3 i[];            };

// clang-format on


#endif  // LAYOUTS_GLSL
