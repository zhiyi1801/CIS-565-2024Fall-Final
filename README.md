ReSTIR with DirectX Raytracing
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Zhiyi Zhou
  * [[GitHub](https://github.com/zhiyi1801)] [[LinkedIn](https://www.linkedin.com/in/%E5%B8%9C%E4%B8%80-%E5%91%A8-10548328b/)].
* Jichu Mao
  * [[GitHub](https://github.com/jichumao)] [[LinkedIn](https://www.linkedin.com/in/jichumao/)].

## Introduction

This repository contains our implementation of [**World-Space Spatiotemporal Path Resampling for Path Tracing**](https://wangningbei.github.io/2023/ReSTIR.html) with Vulkan Raytracing. Our base code is from [Nvidia's Vulkan raytrace renderer](https://github.com/nvpro-samples/vk_raytrace/tree/master). 

![](imgs/team-9.jpg)

The rendering pipeline can be switched from:
* **Ray Tracing Pipeline**: RayGen, Closest-Hit, Miss, Any-Hit model
* **Ray Query**: Compute shader using Ray Queries
* **ReSTIR**: Our World Space ReSTIR implementation.


## Setup

~~~~ 
git clone https://github.com/zhiyi1801/CIS-565-2024Fall-Final.git
cd ./CIS-565-2024Fall-Final
mkdir build
cd ./build
cmake ../
~~~~

- Unlike the original Nvidia's Vulkan raytrace renderer, which requires you to separately clone the nvpro_core library and the renderer itself, we’ve simplified the process by bundling these dependencies directly into this repository. This eliminates the need for additional steps—just clone and build.

- We recommend to use Visual Studio 2022 to develop this project.

### Controls

| Action | Description |
|--------|-------------|
|`LMB`        | Rotate around the target|
|`RMB`        | Dolly in/out|
|`MMB`        | Pan along view plane|
|`LMB + Shift`| Dolly in/out|
|`LMB + Ctrl` | Pan |
|`LMB + Alt`  | Look around |
|`Mouse wheel`| Dolly in/out |
|`Mouse wheel + Shift`| Zoom in/out (FOV)
|`Space`| Set interest point on the surface under the mouse cursor.
|`F10`| Toggle UI pane.
|Drag and drop glTF files (`.gltf` or `.glb`) into viewer|Change glTF model
|Drag and drop HDR files (`.hdr`) into viewer|Change HDR lighting


## Performance Analysis
To be finished

## Timeline

- [Pitch](imgs/Pitch.pdf) 
- [Milestone 1(Nov 13)](imgs/Milestone1.pdf)
  - Basic Vulkan Ray-Tracing Pipeline Setup in Vulkan
  - Light source importance sampling for both environment map and mesh lights
  - Hash Grid Data Structure Setup
  - Research on RIS(Resampled Importance Sampling), Reservoir-based sample Algorithm, and Denoise techs

- [Milestone 2(Nov 25)](imgs/Milestone2.pdf)
  - Completed hash grid Construction & Visualization
  - Completed World Space ReSTIR Direct illumination
  - Completed basic Open Image Denoiser integration

- [Milestone 3(Dec 02)](imgs/Milestone3.pdf)
  - Completed basic Spatial Reuse
  - Fixed bugs for Temporal Reuse 
  - Fixed several bugs for DI
  - Project refactoring and code optimization 


## References
* [glTF format specification](https://github.com/KhronosGroup/glTF)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF-Sample-Models)
* [tiny glTF library](https://github.com/syoyo/tinygltf)
* [Ray Tracer Tutorial](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR)
* [Vulkan ray tracing](https://www.khronos.org/blog/vulkan-ray-tracing-final-specification-release)
* [glTF 2.0](https://www.khronos.org/gltf/)
* [ray tracing tutorial](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR)