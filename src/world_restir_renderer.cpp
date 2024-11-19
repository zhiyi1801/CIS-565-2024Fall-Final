#include "nvh/alignment.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "world_restir_renderer.hpp"
#include "tools.hpp"
#include <spirv_cross/spirv_cross.hpp>
#include <iostream>

#include "autogen/pathtrace.comp.h"
#include "autogen/pathtrace.rahit.h"
#include "autogen/pathtrace.rchit.h"
#include "autogen/pathtrace.rgen.h"
#include "autogen/pathtrace.rmiss.h"
#include "autogen/pathtraceShadow.rmiss.h"
#include "autogen/ReflectTypes.comp.h"
#include "autogen/pathtrace_metallicworkflow.comp.h"
#include "autogen/gbufferPass.comp.h"
#include "autogen/initial_ray_trace_pass.comp.h"
#include "autogen/init_reservoir.comp.h"
#include "autogen/build_hash_grid.comp.h"
#include "autogen/scan_cell.comp.h"

VkPipeline createComputePipeline(VkDevice device, VkComputePipelineCreateInfo createInfo, const uint32_t* shader, size_t bytes) {
	VkPipeline pipeline;
	auto shaderModule = nvvk::createShaderModule(device, shader, bytes);
	createInfo.stage.module = shaderModule;
	vkCreateComputePipelines(device, {}, 1, &createInfo, nullptr, &pipeline);
	vkDestroyShaderModule(device, shaderModule, nullptr);
	return pipeline;
}

void WorldRestirRenderer::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
	m_device = device;
	m_pAlloc = allocator;
	m_queueIndex = familyIndex;
	m_debug.setup(device);
}

void WorldRestirRenderer::create(const VkExtent2D& size, std::vector<VkDescriptorSetLayout>& rtDescSetLayouts, Scene* scene)
{
	m_size = size;
	MilliTimer timer;
	LOGI("Create ReSTIR Pipeline");

	std::vector<VkPushConstantRange> push_constants;
	push_constants.push_back({ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState) });

	// Create Gbuffer and Textures
	createImage();
	createBuffer();

	createDescriptorSet();
	rtDescSetLayouts.push_back(m_descSetLayout);

	VkPipelineLayoutCreateInfo layout_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	layout_info.pushConstantRangeCount = static_cast<uint32_t>(push_constants.size());
	layout_info.pPushConstantRanges = push_constants.data();
	layout_info.setLayoutCount = static_cast<uint32_t>(rtDescSetLayouts.size());
	layout_info.pSetLayouts = rtDescSetLayouts.data();
	vkCreatePipelineLayout(m_device, &layout_info, nullptr, &m_pipelineLayout);

	VkComputePipelineCreateInfo createInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	createInfo.layout = m_pipelineLayout;
	createInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	createInfo.stage.pName = "main";

	m_pipeline = createComputePipeline(m_device, createInfo, pathtrace_metallicworkflow_comp, sizeof(pathtrace_metallicworkflow_comp));
	m_debug.setObjectName(m_pipeline, "Test");

	m_InitialSamplePipeline = createComputePipeline(m_device, createInfo, initial_ray_trace_pass_comp, sizeof(initial_ray_trace_pass_comp));
	m_debug.setObjectName(m_InitialSamplePipeline, "Initial Sample");

	m_GbufferPipeline = createComputePipeline(m_device, createInfo, gbufferPass_comp, sizeof(gbufferPass_comp));
	m_debug.setObjectName(m_GbufferPipeline, "Gbuffer");

	m_InitialReservoirPipeline = createComputePipeline(m_device, createInfo, init_reservoir_comp, sizeof(init_reservoir_comp));
	m_debug.setObjectName(m_InitialReservoirPipeline, "Initial Reservoir");

	m_BuildHashGridPipeline = createComputePipeline(m_device, createInfo, build_hash_grid_comp, sizeof(build_hash_grid_comp));
	m_debug.setObjectName(m_BuildHashGridPipeline, "Build Hash Grid");

	m_ScanCellPipeline = createComputePipeline(m_device, createInfo, scan_cell_comp, sizeof(scan_cell_comp));
	m_debug.setObjectName(m_ScanCellPipeline, "Scan Cell");

	timer.print();
}

void WorldRestirRenderer::update(const VkExtent2D& size)
{
	m_size = size;
	m_pAlloc->destroy(m_FinalSample);
	m_pAlloc->destroy(m_InitialReservoir);
	m_pAlloc->destroy(m_AppendBuffer);
	m_pAlloc->destroy(m_testImage);
	m_pAlloc->destroy(m_testUintImage);
	m_pAlloc->destroy(m_InitialSamples);
	m_pAlloc->destroy(m_ReconnectionData);
	for (int i = 0; i < 2; ++i)
	{
		m_pAlloc->destroy(m_gbuffer[i]);
		m_pAlloc->destroy(m_Reservoirs[i]);
		m_pAlloc->destroy(m_CellStorage[i]);
		m_pAlloc->destroy(m_IndexBuffer[i]);
		m_pAlloc->destroy(m_CheckSumBuffer[i]);
		m_pAlloc->destroy(m_CellCounter[i]);
	}
	createImage();
	createBuffer();
	updateDescriptorSet();
}

void WorldRestirRenderer::destroy()
{
	m_pAlloc->destroy(m_InitialReservoir);
	m_pAlloc->destroy(m_AppendBuffer);
	m_pAlloc->destroy(m_InitialSamples);
	m_pAlloc->destroy(m_ReconnectionData);
	m_pAlloc->destroy(m_FinalSample);
	m_pAlloc->destroy(m_testImage);
	m_pAlloc->destroy(m_testUintImage);

	m_pAlloc->destroy(m_gbuffer[0]);
	m_pAlloc->destroy(m_gbuffer[1]);
	m_pAlloc->destroy(m_Reservoirs[0]);
	m_pAlloc->destroy(m_Reservoirs[1]);
	m_pAlloc->destroy(m_CellStorage[0]);
	m_pAlloc->destroy(m_CellStorage[1]);
	m_pAlloc->destroy(m_IndexBuffer[0]);
	m_pAlloc->destroy(m_IndexBuffer[1]);
	m_pAlloc->destroy(m_CheckSumBuffer[0]);
	m_pAlloc->destroy(m_CheckSumBuffer[1]);
	m_pAlloc->destroy(m_CellCounter[0]);
	m_pAlloc->destroy(m_CellCounter[1]);

	vkFreeDescriptorSets(m_device, m_descPool, static_cast<uint32_t>(m_descSet.size()), m_descSet.data());
	vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

	auto destroyPipeline = [&](VkPipeline& pipeline) {
		vkDestroyPipeline(m_device, pipeline, nullptr);
		pipeline = VK_NULL_HANDLE;
		};
	destroyPipeline(m_pipeline);
	destroyPipeline(m_InitialSamplePipeline);
	destroyPipeline(m_GbufferPipeline);
	destroyPipeline(m_InitialReservoirPipeline);
	destroyPipeline(m_BuildHashGridPipeline);
	destroyPipeline(m_ScanCellPipeline);

	vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
	m_pipelineLayout = VK_NULL_HANDLE;
	m_pipeline = VK_NULL_HANDLE;
	m_GbufferPipeline = VK_NULL_HANDLE;
	m_InitialSamplePipeline = VK_NULL_HANDLE;
	m_BuildHashGridPipeline = VK_NULL_HANDLE;
}

void WorldRestirRenderer::createBuffer()
{
	VkDeviceSize fullScreenSize = m_size.width * m_size.height;
	VkDeviceSize halfScreenSize = (1.0f/2.0f * m_size.width) * (1.0f/2.0f * m_size.height);

	VkDeviceSize elementCount = fullScreenSize;

	m_FinalSample = m_pAlloc->createBuffer(elementCount * sizeof(FinalSample), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_InitialReservoir = m_pAlloc->createBuffer(elementCount * sizeof(Reservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_AppendBuffer = m_pAlloc->createBuffer(elementCount * sizeof(HashAppendData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_InitialSamples = m_pAlloc->createBuffer(elementCount * sizeof(InitialSample), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_ReconnectionData = m_pAlloc->createBuffer(elementCount * sizeof(ReconnectionData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	VkDeviceSize reseviorCount = 2 * elementCount;

	VkDeviceSize hashBufferSize = 3200 * sizeof(uint32_t);
	for (int i = 0; i < 2; ++i)
	{
		m_Reservoirs[i] = m_pAlloc->createBuffer(reseviorCount * sizeof(Reservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_CellStorage[i] = m_pAlloc->createBuffer(elementCount * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_IndexBuffer[i] = m_pAlloc->createBuffer(hashBufferSize * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
		m_CheckSumBuffer[i] = m_pAlloc->createBuffer(hashBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_CellCounter[i] = m_pAlloc->createBuffer(hashBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}
}

void WorldRestirRenderer::createImage()
{
	// Creating the color image
	{
		auto colorCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_gbufferFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

		auto testImageCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_testImageFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

		auto testUintImageCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_testUintImageFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

		nvvk::Image gbimage1 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(gbimage1.image);
		nvvk::Image gbimage2 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(gbimage2.image);
		VkImageViewCreateInfo ivInfo1 = nvvk::makeImageViewCreateInfo(gbimage1.image, colorCreateInfo);
		VkImageViewCreateInfo ivInfo2 = nvvk::makeImageViewCreateInfo(gbimage2.image, colorCreateInfo);

		nvvk::Image testImage = m_pAlloc->createImage(testImageCreateInfo);
		NAME_VK(testImage.image);
		VkImageViewCreateInfo ivInfoTest = nvvk::makeImageViewCreateInfo(testImage.image, testImageCreateInfo);

		nvvk::Image testUintImage = m_pAlloc->createImage(testUintImageCreateInfo);
		NAME_VK(testUintImage.image);
		VkImageViewCreateInfo ivInfoUintTest = nvvk::makeImageViewCreateInfo(testUintImage.image, testUintImageCreateInfo);

		m_gbuffer[0] = m_pAlloc->createTexture(gbimage1, ivInfo1);
		m_gbuffer[1] = m_pAlloc->createTexture(gbimage2, ivInfo2);
		m_gbuffer[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_gbuffer[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		m_testImage = m_pAlloc->createTexture(testImage, ivInfoTest);
		m_testImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		m_testUintImage = m_pAlloc->createTexture(testUintImage, ivInfoUintTest);
		m_testUintImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	// Setting the image layout for both color and depth
	{
		nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
		auto              cmdBuf = genCmdBuf.createCommandBuffer();
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_testImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_testUintImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		genCmdBuf.submitAndWait(cmdBuf);
	}
}

void WorldRestirRenderer::createDescriptorSet()
{
	m_bind = nvvk::DescriptorSetBindings{};

	VkShaderStageFlags flag = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
		| VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	m_bind.addBinding({ ReSTIRBindings::eLastGbuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCurrentGbuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	m_bind.addBinding({ ReSTIRBindings::eInitialReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eAppend, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eFinal, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCell, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eIndex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCheckSum, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCellCounter, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eInitialSamples, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eReconnection, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_bind.addBinding({ ReSTIRBindings::eTestUintImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eTestImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	m_descPool = m_bind.createPool(m_device, m_descSet.size(), VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
	CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
	CREATE_NAMED_VK(m_descSet[0], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	CREATE_NAMED_VK(m_descSet[1], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

	updateDescriptorSet();
}

void WorldRestirRenderer::updateDescriptorSet()
{
	std::array<VkWriteDescriptorSet, 14> writes;
	VkDeviceSize fullScreenSize = m_size.width * m_size.height;
	VkDeviceSize elementCount = fullScreenSize;
	VkDeviceSize hashBufferSize = 3200 * sizeof(uint32_t);
	VkDeviceSize reseviorCount = 2 * elementCount;

	for (int i = 0; i < 2; i++) {
		VkDescriptorBufferInfo initialResvervoirBufInfo = { m_Reservoirs[i].buffer, 0, elementCount * sizeof(Reservoir) };
		VkDescriptorBufferInfo reservoirBufInfo = { m_Reservoirs[i].buffer, 0, reseviorCount * sizeof(Reservoir)};
		VkDescriptorBufferInfo appendBufInfo = { m_AppendBuffer.buffer, 0, elementCount * sizeof(HashAppendData) };
		VkDescriptorBufferInfo finalSampleBufInfo = { m_FinalSample.buffer, 0, elementCount * sizeof(FinalSample) };
		VkDescriptorBufferInfo initialSampleBufInfo = { m_InitialSamples.buffer, 0, elementCount * sizeof(InitialSample) };
		VkDescriptorBufferInfo reconnectionBufInfo = { m_ReconnectionData.buffer, 0, elementCount * sizeof(ReconnectionData) };

		VkDescriptorBufferInfo cellStorageBufInfo = { m_CellStorage[i].buffer, 0, elementCount * sizeof(uint) };
		VkDescriptorBufferInfo indexeBufInfo = { m_IndexBuffer[i].buffer, 0, hashBufferSize * 4};
		VkDescriptorBufferInfo checkSumBufInfo = { m_CheckSumBuffer[i].buffer, 0, hashBufferSize };
		VkDescriptorBufferInfo cellCounterBufInfo = { m_CellCounter[i].buffer, 0, hashBufferSize };

		writes[0] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eLastGbuffer, &m_gbuffer[i].descriptor);
		writes[1] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCurrentGbuffer, &m_gbuffer[!i].descriptor);

		writes[2] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eInitialReservoirs, &initialResvervoirBufInfo);
		writes[3] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eReservoirs, &reservoirBufInfo);
		writes[4] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eAppend, &appendBufInfo);
		writes[5] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eFinal, &finalSampleBufInfo);
		writes[6] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCell, &cellStorageBufInfo);
		writes[7] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eIndex, &indexeBufInfo);
		writes[8] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCheckSum, &checkSumBufInfo);
		writes[9] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCellCounter, &cellCounterBufInfo);
		writes[10] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eInitialSamples, &initialSampleBufInfo);
		writes[11] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eReconnection, &reconnectionBufInfo);

		writes[12] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eTestUintImage, &m_testUintImage.descriptor);
		writes[13] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eTestImage, &m_testImage.descriptor);

		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}
}

void InsertPerfMarker(VkCommandBuffer commandBuffer, const char* name, float color[4]) {
	VkDebugUtilsLabelEXT labelInfo = {};
	labelInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	labelInfo.pLabelName = name;
	labelInfo.color[0] = color[0];
	labelInfo.color[1] = color[1];
	labelInfo.color[2] = color[2];
	labelInfo.color[3] = color[3];

	vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &labelInfo);
}

void EndPerfMarker(VkCommandBuffer commandBuffer) {
	vkCmdEndDebugUtilsLabelEXT(commandBuffer);
}

// Floor of log2 of x
inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

// Ceiling of log2 of x
inline int ilog2ceil(int x) {
	return x == 1 ? 0 : ilog2(x - 1) + 1;
}

#define GROUP_SIZE 8  // Same group size as in compute shader

// Exclusive scan
void WorldRestirRenderer::cellScan(const VkCommandBuffer& cmdBuf)
{
	// Iteration of scan
	uint ilogn = ilog2ceil(m_state.cellCount);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScanCellPipeline);
	for (int i = 0; i < ilogn; ++i)
	{
		m_state.cellScanIte = i;
		vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &m_state);
		vkCmdDispatch(cmdBuf, (m_state.cellCount + (1024 - 1)) / 1024, 1, 1);
	}
}

void WorldRestirRenderer::run(const VkCommandBuffer& cmdBuf, const VkExtent2D& size, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet>& descSets, uint frames)
{
	descSets.push_back(m_descSet[(frames + 1) % 2]);
	// Preparing for the compute shader
	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
		static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);

	// Set buffers to zero
	VkDeviceSize cellSize = 3200;
	VkDeviceSize hashBufferSize = cellSize * sizeof(uint32_t);
	vkCmdFillBuffer(cmdBuf, m_CellCounter[(frames + 1) % 2].buffer, 0, hashBufferSize, 0);
	vkCmdFillBuffer(cmdBuf, m_CheckSumBuffer[(frames + 1) % 2].buffer, 0, hashBufferSize, 0);
	vkCmdFillBuffer(cmdBuf, m_IndexBuffer[(frames + 1) % 2].buffer, 0, hashBufferSize * 4, 0);

	// Insert a barrier
	VkBufferMemoryBarrier bufferBarriers[3] = {};

	// Barrier for m_CellCounter
	bufferBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	bufferBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	bufferBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarriers[0].buffer = m_CellCounter[(frames + 1) % 2].buffer;
	bufferBarriers[0].offset = 0;
	bufferBarriers[0].size = hashBufferSize;

	// Barrier for m_CheckSumBuffer
	bufferBarriers[1] = bufferBarriers[0];
	bufferBarriers[1].buffer = m_CheckSumBuffer[(frames + 1) % 2].buffer;

	// Barrier for m_IndexBuffer
	bufferBarriers[2] = bufferBarriers[0];
	bufferBarriers[2].buffer = m_IndexBuffer[(frames + 1) % 2].buffer;
	bufferBarriers[2].size = hashBufferSize * 4;

	// Apply the pipeline barrier
	vkCmdPipelineBarrier(
		cmdBuf,
		VK_PIPELINE_STAGE_TRANSFER_BIT,                  // Source stage (Fill Buffer)
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,            // Destination stage (Compute Shader)
		0,                                              // No flags
		0, nullptr,                                     // No memory barriers
		3, bufferBarriers,                              // Buffer barriers
		0, nullptr                                      // No image barriers
	);

	// Sending the push constant information
	// TODO
	// The number of cells(3200000)
	m_state.cellCount = cellSize;

	// The iteration counter of scan
	m_state.cellScanIte = 0;
	m_state.environmentProb = 0.5;
	m_state.maxBounces = 3;
	vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &m_state);
	
	float color[3][4] = { {0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f} };
	uint count = 0;
	// Dispatching the shader
	InsertPerfMarker(cmdBuf, "Compute Shader: GBuffer", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_GbufferPipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);

	InsertPerfMarker(cmdBuf, "Compute Shader: Initial Sample", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_InitialSamplePipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);

	InsertPerfMarker(cmdBuf, "Compute Shader: Initial Reservoir", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_InitialReservoirPipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);

	InsertPerfMarker(cmdBuf, "Compute Shader: Cell Scan", color[(count++) % 3]);
	//vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScanCellPipeline);
	//vkCmdDispatch(cmdBuf, (cellSize + (GROUP_SIZE - 1)) / GROUP_SIZE, 1, 1);
	this->cellScan(cmdBuf);
	EndPerfMarker(cmdBuf);

	InsertPerfMarker(cmdBuf, "Compute Shader: Build Hash Grid", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildHashGridPipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);
}

//void reflectTest()
//{
//	const size_t arraySize = sizeof(ReflectTypes_comp) / sizeof(ReflectTypes_comp[0]);
//	std::vector<uint32_t> myVector(ReflectTypes_comp, ReflectTypes_comp + arraySize);
//
//	spirv_cross::Compiler comp(myVector);
//	spirv_cross::ShaderResources resources = comp.get_shader_resources();
//
//	for (auto& resource : resources.storage_buffers) {
//		const auto& type = comp.get_type(resource.base_type_id);
//		unsigned memberCount = type.member_types.size();
//
//		std::cout << "Storage buffer: " << resource.name << std::endl;
//		for (unsigned i = 0; i < memberCount; ++i) {
//			const auto& memberType = comp.get_type(type.member_types[i]);
//			std::string name = comp.get_member_name(type.self, i);
//			size_t size = comp.get_declared_struct_member_size(type, i);
//
//			std::cout << "  Member name: " << name << ", size: " << size << std::endl;
//		}
//	}
//}
//
//void WorldRestirRenderer::test()
//{
//	reflectTest();
//}