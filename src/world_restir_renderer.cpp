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
#include "autogen/scan_validate.comp.h"
#include "autogen/spatial_temporal_resampling.comp.h"
#include "autogen//final_sample.comp.h"
#include "autogen/final_shading.comp.h"
#include "autogen/ReSTIR_DI.comp.h"
#include "autogen/ReSTIR_Indirect.comp.h"

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
	m_CellSize = cellSizeNoHash * 32;
	m_DebugBufferSize = 1000;
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

	m_ScanCellValidationPipeline = createComputePipeline(m_device, createInfo, scan_validate_comp, sizeof(scan_validate_comp));
	m_debug.setObjectName(m_ScanCellPipeline, "Scan Cell Validate");

	m_STResamplePipeline = createComputePipeline(m_device, createInfo, spatial_temporal_resampling_comp, sizeof(spatial_temporal_resampling_comp));
	m_debug.setObjectName(m_ScanCellPipeline, "Spatial Temporal Reuse");

	m_FinalSamplePipeline = createComputePipeline(m_device, createInfo, final_sample_comp, sizeof(final_sample_comp));
	m_debug.setObjectName(m_FinalSamplePipeline, "Final Sample");

	m_FinalShadingPipeline = createComputePipeline(m_device, createInfo, final_shading_comp, sizeof(final_shading_comp));
	m_debug.setObjectName(m_FinalShadingPipeline, "Final Shading");

	m_DirectLightPipeline = createComputePipeline(m_device, createInfo, ReSTIR_DI_comp, sizeof(ReSTIR_DI_comp));
	m_debug.setObjectName(m_DirectLightPipeline, "Direct Light");

	m_IndirectPipeline = createComputePipeline(m_device, createInfo, ReSTIR_Indirect_comp, sizeof(ReSTIR_Indirect_comp));
	m_debug.setObjectName(m_IndirectPipeline, "Indirect Light");

	timer.print();
}

void WorldRestirRenderer::update(const VkExtent2D& size)
{
	m_size = size;
	m_pAlloc->destroy(m_FinalSample);
	m_pAlloc->destroy(m_InitialReservoir);
	m_pAlloc->destroy(m_AppendBuffer);
	m_pAlloc->destroy(m_DebugImage);
	m_pAlloc->destroy(m_DebugUintImage);
	m_pAlloc->destroy(m_InitialSamples);
	m_pAlloc->destroy(m_ReconnectionData);
	m_pAlloc->destroy(m_IndexTempBuffer);
	m_pAlloc->destroy(m_DebugUintBuffer);
	m_pAlloc->destroy(m_DebugFloatBuffer);
	m_pAlloc->destroy(m_motionVector);

	for (int i = 0; i < 2; ++i)
	{
		m_pAlloc->destroy(m_gbuffer[i]);
		m_pAlloc->destroy(m_Reservoirs[i]);
		m_pAlloc->destroy(m_CellStorage[i]);
		m_pAlloc->destroy(m_IndexBuffer[i]);
		m_pAlloc->destroy(m_CheckSumBuffer[i]);
		m_pAlloc->destroy(m_CellCounter[i]);
		m_pAlloc->destroy(m_DirectReservoir[i]);
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
	m_pAlloc->destroy(m_DebugImage);
	m_pAlloc->destroy(m_DebugUintImage);
	m_pAlloc->destroy(m_IndexTempBuffer);
	m_pAlloc->destroy(m_DebugUintBuffer);
	m_pAlloc->destroy(m_DebugFloatBuffer);
	m_pAlloc->destroy(m_motionVector);

	for (int i = 0; i < 2; ++i)
	{
		m_pAlloc->destroy(m_gbuffer[i]);
		m_pAlloc->destroy(m_Reservoirs[i]);
		m_pAlloc->destroy(m_CellStorage[i]);
		m_pAlloc->destroy(m_IndexBuffer[i]);
		m_pAlloc->destroy(m_CheckSumBuffer[i]);
		m_pAlloc->destroy(m_CellCounter[i]);
		m_pAlloc->destroy(m_DirectReservoir[i]);
	}

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
	destroyPipeline(m_ScanCellValidationPipeline);
	destroyPipeline(m_STResamplePipeline);

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

	VkDeviceSize hashBufferSize = m_CellSize * sizeof(uint32_t);
	// index temp buffer for cell index scan
	m_IndexTempBuffer = m_pAlloc->createBuffer(hashBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	// Debug buffers
	m_DebugUintBuffer = m_pAlloc->createBuffer(m_DebugBufferSize * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	m_DebugFloatBuffer = m_pAlloc->createBuffer(m_DebugBufferSize * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	VkDeviceSize directSize = m_size.width * m_size.height * sizeof(DirectReservoir);
	VkDeviceSize indirectSize = m_size.width * m_size.height * sizeof(IndirectReservoir);
	for (int i = 0; i < 2; ++i)
	{
		m_Reservoirs[i] = m_pAlloc->createBuffer(reseviorCount * sizeof(Reservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_CellStorage[i] = m_pAlloc->createBuffer(elementCount * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_IndexBuffer[i] = m_pAlloc->createBuffer(hashBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		m_CheckSumBuffer[i] = m_pAlloc->createBuffer(hashBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_CellCounter[i] = m_pAlloc->createBuffer(hashBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_DirectReservoir[i] = m_pAlloc->createBuffer(directSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_IndirectReservoir[i] = m_pAlloc->createBuffer(indirectSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}
}

void WorldRestirRenderer::createImage()
{
	// Creating the color image
	{
		auto colorCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_gbufferFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

		auto DebugImageCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_DebugImageFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

		auto DebugUintImageCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, m_DebugUintImageFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);

		nvvk::Image gbimage1 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(gbimage1.image);
		nvvk::Image gbimage2 = m_pAlloc->createImage(colorCreateInfo);
		NAME_VK(gbimage2.image);
		VkImageViewCreateInfo ivInfo1 = nvvk::makeImageViewCreateInfo(gbimage1.image, colorCreateInfo);
		VkImageViewCreateInfo ivInfo2 = nvvk::makeImageViewCreateInfo(gbimage2.image, colorCreateInfo);

		nvvk::Image DebugImage = m_pAlloc->createImage(DebugImageCreateInfo);
		NAME_VK(DebugImage.image);
		VkImageViewCreateInfo ivInfoDebug = nvvk::makeImageViewCreateInfo(DebugImage.image, DebugImageCreateInfo);

		nvvk::Image DebugUintImage = m_pAlloc->createImage(DebugUintImageCreateInfo);
		NAME_VK(DebugUintImage.image);
		VkImageViewCreateInfo ivInfoUintDebug = nvvk::makeImageViewCreateInfo(DebugUintImage.image, DebugUintImageCreateInfo);

		auto motionVecCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_motionVectorFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
		nvvk::Image motionVecImg = m_pAlloc->createImage(motionVecCreateInfo);
		NAME_VK(motionVecImg.image);

		VkImageViewCreateInfo mvivInfo = nvvk::makeImageViewCreateInfo(motionVecImg.image, motionVecCreateInfo);
		m_motionVector = m_pAlloc->createTexture(motionVecImg, mvivInfo);
		m_motionVector.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		m_gbuffer[0] = m_pAlloc->createTexture(gbimage1, ivInfo1);
		m_gbuffer[1] = m_pAlloc->createTexture(gbimage2, ivInfo2);
		m_gbuffer[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_gbuffer[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		m_DebugImage = m_pAlloc->createTexture(DebugImage, ivInfoDebug);
		m_DebugImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		m_DebugUintImage = m_pAlloc->createTexture(DebugUintImage, ivInfoUintDebug);
		m_DebugUintImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	// Setting the image layout for both color and depth
	{
		nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
		auto              cmdBuf = genCmdBuf.createCommandBuffer();
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_DebugImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_DebugUintImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_motionVector.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

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
	m_bind.addBinding({ ReSTIRBindings::eMotionVector, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });

	// Direct Light Reservoirs
	m_bind.addBinding({ ReSTIRBindings::ePrevDirectReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCurrentDirectReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	// Indirect Light Reservoirs
	m_bind.addBinding({ ReSTIRBindings::ePrevIndirectReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCurrentIndirectReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_bind.addBinding({ ReSTIRBindings::eInitialReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCurrentReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::ePrevReservoirs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eAppend, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eFinal, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCell, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eIndex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCheckSum, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eCellCounter, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eInitialSamples, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eReconnection, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eIndexTemp, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	// Debug images
	m_bind.addBinding({ ReSTIRBindings::eDebugUintImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eDebugImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	// Debug buffers
	m_bind.addBinding({ ReSTIRBindings::eDebugUintBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eDebugFloatBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	m_descPool = m_bind.createPool(m_device, m_descSet.size(), VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
	CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
	CREATE_NAMED_VK(m_descSet[0], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	CREATE_NAMED_VK(m_descSet[1], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

	updateDescriptorSet();
}

void WorldRestirRenderer::updateDescriptorSet()
{
	std::vector<VkWriteDescriptorSet> writes;
	VkDeviceSize fullScreenSize = m_size.width * m_size.height;
	VkDeviceSize elementCount = fullScreenSize;
	VkDeviceSize hashBufferSize = m_CellSize * sizeof(uint32_t);
	VkDeviceSize reseviorCount = 2 * elementCount;
	VkDeviceSize directResvSize = m_size.width * m_size.height * sizeof(DirectReservoir);
	VkDeviceSize indirectResvSize = m_size.width * m_size.height * sizeof(IndirectReservoir);

	for (int i = 0; i < 2; i++) {
		writes.clear();
		VkDescriptorBufferInfo initialResvervoirBufInfo = { m_InitialReservoir.buffer, 0, elementCount * sizeof(Reservoir) };
		VkDescriptorBufferInfo currentReservoirBufInfo = { m_Reservoirs[i].buffer, 0, reseviorCount * sizeof(Reservoir)};
		VkDescriptorBufferInfo prevReservoirBufInfo = { m_Reservoirs[!i].buffer, 0, reseviorCount * sizeof(Reservoir)};
		VkDescriptorBufferInfo appendBufInfo = { m_AppendBuffer.buffer, 0, elementCount * sizeof(HashAppendData) };
		VkDescriptorBufferInfo finalSampleBufInfo = { m_FinalSample.buffer, 0, elementCount * sizeof(FinalSample) };
		VkDescriptorBufferInfo initialSampleBufInfo = { m_InitialSamples.buffer, 0, elementCount * sizeof(InitialSample) };
		VkDescriptorBufferInfo reconnectionBufInfo = { m_ReconnectionData.buffer, 0, elementCount * sizeof(ReconnectionData) };

		// Direct Light Reservoirs
		VkDescriptorBufferInfo lastDirectResvBufInfo = { m_DirectReservoir[!i].buffer, 0, directResvSize };
		VkDescriptorBufferInfo thisDirectResvBufInfo = { m_DirectReservoir[i].buffer, 0, directResvSize };

		// Indirect Light Reservoirs
		VkDescriptorBufferInfo lastIndirectResvBufInfo = { m_IndirectReservoir[!i].buffer, 0, indirectResvSize };
		VkDescriptorBufferInfo thisIndirectResvBufInfo = { m_IndirectReservoir[i].buffer, 0, indirectResvSize };

		VkDescriptorBufferInfo cellStorageBufInfo = { m_CellStorage[i].buffer, 0, elementCount * sizeof(uint) };
		VkDescriptorBufferInfo indexBufInfo = { m_IndexBuffer[i].buffer, 0, hashBufferSize};
		VkDescriptorBufferInfo indexTempBufInfo = { m_IndexTempBuffer.buffer, 0, hashBufferSize};
		VkDescriptorBufferInfo checkSumBufInfo = { m_CheckSumBuffer[i].buffer, 0, hashBufferSize };
		VkDescriptorBufferInfo cellCounterBufInfo = { m_CellCounter[i].buffer, 0, hashBufferSize };

		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eLastGbuffer, &m_gbuffer[i].descriptor));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCurrentGbuffer, &m_gbuffer[!i].descriptor));

		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eInitialReservoirs, &initialResvervoirBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCurrentReservoirs, &currentReservoirBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::ePrevReservoirs, &prevReservoirBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eAppend, &appendBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eFinal, &finalSampleBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCell, &cellStorageBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eIndex, &indexBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCheckSum, &checkSumBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCellCounter, &cellCounterBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eInitialSamples, &initialSampleBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eReconnection, &reconnectionBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eIndexTemp, &indexTempBufInfo));

		// debug images desc
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eDebugUintImage, &m_DebugUintImage.descriptor));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eDebugImage, &m_DebugImage.descriptor));

		// Motion vec
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eMotionVector, &m_motionVector.descriptor));

		// Direct Light Reservoirs
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::ePrevDirectReservoirs, &lastDirectResvBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCurrentDirectReservoirs, &thisDirectResvBufInfo));

		// Indirect Light Reservoirs
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::ePrevIndirectReservoirs, &lastIndirectResvBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCurrentIndirectReservoirs, &thisIndirectResvBufInfo));

		// debug buffers desc
		VkDescriptorBufferInfo debugUintBufInfo = { m_DebugUintBuffer.buffer, 0, m_DebugBufferSize * sizeof(uint32_t) };
		VkDescriptorBufferInfo debugFloatBufInfo = { m_DebugFloatBuffer.buffer, 0, m_DebugBufferSize * sizeof(float) };
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eDebugUintBuffer, &debugUintBufInfo));
		writes.emplace_back(m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eDebugFloatBuffer, &debugFloatBufInfo));

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

// Inclusive scan
void WorldRestirRenderer::cellScan(const VkCommandBuffer& cmdBuf, const int frames)
{
	// Iteration of scan
	uint ilogn = ilog2ceil(m_state.cellCount);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScanCellPipeline);
	for (int i = 0; i < ilogn; ++i)
	{
		m_state.cellScanIte = i;
		vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &m_state);
		vkCmdDispatch(cmdBuf, (m_state.cellCount + (1024 - 1)) / 1024, 1, 1);

		// Insert Pipeline Barrier
		VkBufferMemoryBarrier bufferBarriers[2] = {};

		// Barrier for main buffer
		bufferBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bufferBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarriers[0].buffer = m_IndexBuffer[(frames + 1) % 2].buffer;  // �� buffer
		bufferBarriers[0].offset = 0;
		bufferBarriers[0].size = VK_WHOLE_SIZE;

		// Barrier for temp buffer
		bufferBarriers[1] = bufferBarriers[0];
		bufferBarriers[1].buffer = m_IndexTempBuffer.buffer;  // ��ʱ buffer

		vkCmdPipelineBarrier(
			cmdBuf,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // 
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,  // 
			0,                                    // 
			0, nullptr,                           // Memory Barriers
			2, bufferBarriers,                    // Buffer Barriers
			0, nullptr                            // Image Barriers
		);
	}
	if (( (ilogn - 1) & 1) == 1)
	{
		// Set copy region
		VkBufferCopy copyRegion = {};
		copyRegion.srcOffset = 0;          
		copyRegion.dstOffset = 0;           
		copyRegion.size = m_CellSize * sizeof(uint);    

		vkCmdCopyBuffer(
			cmdBuf,  // cmd buffer
			m_IndexTempBuffer.buffer,      // source buffer
			m_IndexBuffer[(frames + 1) % 2].buffer,      // destination buffer
			1,              // number of copy regions
			&copyRegion     // copy regions
		);

		// Insert Barrier
		VkBufferMemoryBarrier bufferBarrier = {};
		bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT; 
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.buffer = m_IndexBuffer[(frames + 1) % 2].buffer;
		bufferBarrier.offset = 0;
		bufferBarrier.size = VK_WHOLE_SIZE;

		vkCmdPipelineBarrier(
			cmdBuf,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr
		);
	}
}

void WorldRestirRenderer::run(const VkCommandBuffer& cmdBuf, const VkExtent2D& size, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet>& descSets, uint frames)
{
	VkDeviceSize fullScreenSize = m_size.width * m_size.height;
	VkDeviceSize halfScreenSize = (1.0f / 2.0f * m_size.width) * (1.0f / 2.0f * m_size.height);

	VkDeviceSize elementCount = fullScreenSize;
	VkDeviceSize reseviorCount = 2 * elementCount;

	descSets.push_back(m_descSet[(frames + 1) % 2]);
	// Preparing for the compute shader
	vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
		static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);

	// Set buffers to zero
	VkDeviceSize cellSize = m_CellSize;
	VkDeviceSize hashBufferSize = cellSize * sizeof(uint);
	vkCmdFillBuffer(cmdBuf, m_CellCounter[(frames + 1) % 2].buffer, 0, hashBufferSize, 0);
	vkCmdFillBuffer(cmdBuf, m_CheckSumBuffer[(frames + 1) % 2].buffer, 0, hashBufferSize, 0);
	vkCmdFillBuffer(cmdBuf, m_IndexBuffer[(frames + 1) % 2].buffer, 0, hashBufferSize, 0);
	vkCmdFillBuffer(cmdBuf, m_IndexTempBuffer.buffer, 0, hashBufferSize, 0);
	//vkCmdFillBuffer(cmdBuf, m_DebugUintBuffer.buffer, 0, m_DebugBufferSize * sizeof(uint), 0);
	//vkCmdFillBuffer(cmdBuf, m_DebugFloatBuffer.buffer, 0, m_DebugBufferSize * sizeof(float), 0);

	// Clear images
	{
		VkClearColorValue clearColor = { {1.0f, 0.0f, 0.0f, 1.0f} };
		VkImageSubresourceRange range = {};
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		vkCmdClearColorImage(
			cmdBuf,
			m_DebugImage.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			&clearColor,
			1,
			&range
		);

		vkCmdClearColorImage(
			cmdBuf,
			m_DebugUintImage.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			&clearColor,
			1,
			&range
		);
	}

	// Insert a barrier
	VkBufferMemoryBarrier bufferBarriers[4] = {};

	// Barrier for m_CellCounter
	bufferBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;

	bufferBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	bufferBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	bufferBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarriers[0].buffer = m_Reservoirs[(frames + 1) % 2].buffer;
	bufferBarriers[0].offset = 0;
	bufferBarriers[0].size = hashBufferSize;

	// Barrier for m_CheckSumBuffer
	bufferBarriers[1] = bufferBarriers[0];
	bufferBarriers[1].buffer = m_CheckSumBuffer[(frames + 1) % 2].buffer;

	// Barrier for m_IndexBuffer
	bufferBarriers[2] = bufferBarriers[0];
	bufferBarriers[2].buffer = m_IndexBuffer[(frames + 1) % 2].buffer;

	// Barrier for m_IndexTempBuffer
	bufferBarriers[3] = bufferBarriers[0];
	bufferBarriers[3].buffer = m_IndexTempBuffer.buffer;

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
	m_state.RISSampleNum = 4;
	m_state.reservoirClamp = 80;
	vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RtxState), &m_state);
	
	float color[3][4] = { {0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f} };
	uint count = 0;

	// Dispatching the compute shader
	
	// --------------------------------------------
	// ReSTIR DI Pass
	InsertPerfMarker(cmdBuf, "Compute Shader: Direct Light", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_DirectLightPipeline);
	vkCmdDispatch(cmdBuf, (size.width + (RayTraceBlockSizeX - 1)) / RayTraceBlockSizeX, (size.height + (RayTraceBlockSizeY - 1)) / RayTraceBlockSizeY, 1);

	// ReSTIR DI Pass Barrier
	VkBufferMemoryBarrier DiBarriers[2] = {};

	VkDeviceSize directResvSize = m_size.width * m_size.height * sizeof(DirectReservoir);
	DiBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	DiBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	DiBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	DiBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	DiBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	DiBarriers[0].buffer = m_DirectReservoir[(frames + 1) % 2].buffer;
	DiBarriers[0].offset = 0;
	DiBarriers[0].size = directResvSize;

	DiBarriers[1] = bufferBarriers[0];
	DiBarriers[1].buffer = m_DirectReservoir[(frames) % 2].buffer;

	VkImageMemoryBarrier DiImageBarriers[3] = {};
	DiImageBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	DiImageBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL; 
	DiImageBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;  
	DiImageBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; 
	DiImageBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; 
	DiImageBarriers[0].image = m_motionVector.image; 
	DiImageBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; 
	DiImageBarriers[0].subresourceRange.baseMipLevel = 0;
	DiImageBarriers[0].subresourceRange.levelCount = 1;
	DiImageBarriers[0].subresourceRange.baseArrayLayer = 0;
	DiImageBarriers[0].subresourceRange.layerCount = 1;

	DiImageBarriers[1] = DiImageBarriers[0];
	DiImageBarriers[1].image = m_gbuffer[(frames + 1) % 2].image;
	
	VkMemoryBarrier DiMemoryBarrier = {};
	DiMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	DiMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	DiMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	// Apply the pipeline barrier
	vkCmdPipelineBarrier(
		cmdBuf,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,           // Source stage (Compute Shader)
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,           // Destination stage (Compute Shader | Frag shader)
		0,                                              // No flags
		1, &DiMemoryBarrier,                                     // No memory barriers
		2, DiBarriers,									// Buffer barriers
		2, DiImageBarriers                              // image barriers
	);
	EndPerfMarker(cmdBuf);
	
	// --------------------------------------------
	// 
	// Indirect Light Pass
	//InsertPerfMarker(cmdBuf, "Compute Shader: Indirect Light", color[(count++) % 3]);
	//vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_IndirectPipeline);
	//vkCmdDispatch(cmdBuf, (size.width + (RayTraceBlockSizeX - 1)) / RayTraceBlockSizeX, (size.height + (RayTraceBlockSizeY - 1)) / RayTraceBlockSizeY, 1);
	//EndPerfMarker(cmdBuf);

	// --------------------------------------------
	// Initial Sample Pass
	InsertPerfMarker(cmdBuf, "Compute Shader: Initial Sample", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_InitialSamplePipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);

	// Insert barrier for Initial Sample Pass
	VkBufferMemoryBarrier initialSamplePassBarriers[2] = {};

	// Barrier for m_InitialSamples
	initialSamplePassBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	initialSamplePassBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	initialSamplePassBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	initialSamplePassBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	initialSamplePassBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	initialSamplePassBarriers[0].buffer = m_InitialSamples.buffer;
	initialSamplePassBarriers[0].offset = 0;
	initialSamplePassBarriers[0].size = elementCount * sizeof(InitialSample);

	// Barrier for m_ReconnectionData
	initialSamplePassBarriers[1] = bufferBarriers[0];
	initialSamplePassBarriers[1].buffer = m_ReconnectionData.buffer;
	initialSamplePassBarriers[1].size = elementCount * sizeof(ReconnectionData);

	// Apply the pipeline barrier
	vkCmdPipelineBarrier(
		cmdBuf,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                  // Source stage (Fill Buffer)
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,            // Destination stage (Compute Shader)
		0,                                              // No flags
		0, nullptr,                                     // No memory barriers
		2, initialSamplePassBarriers,				// Buffer barriers
		0, nullptr                                      // No image barriers
	);

	// --------------------------------------------
	// Initial Reservoir Pass
	InsertPerfMarker(cmdBuf, "Compute Shader: Initial Reservoir", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_InitialReservoirPipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);

	// Insert barrier for Initial Reservoir Pass
	VkBufferMemoryBarrier initialReservoirPassBarriers[2] = {};

	// Barrier for m_InitialReservoir
	initialReservoirPassBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	initialReservoirPassBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	initialReservoirPassBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	initialReservoirPassBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	initialReservoirPassBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	initialReservoirPassBarriers[0].buffer = m_InitialReservoir.buffer;
	initialReservoirPassBarriers[0].offset = 0;
	initialReservoirPassBarriers[0].size = elementCount * sizeof(Reservoir);

	// Barrier for m_AppendBuffer
	initialReservoirPassBarriers[1] = bufferBarriers[0];
	initialReservoirPassBarriers[1].buffer = m_AppendBuffer.buffer;
	initialReservoirPassBarriers[1].size = elementCount * sizeof(HashAppendData);

	// Apply the pipeline barrier
	vkCmdPipelineBarrier(
		cmdBuf,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                  // Source stage (Fill Buffer)
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,            // Destination stage (Compute Shader)
		0,                                              // No flags
		0, nullptr,                                     // No memory barriers
		2, initialReservoirPassBarriers,				// Buffer barriers
		0, nullptr                                      // No image barriers
	);

	// --------------------------------------------
	// Cell Scan Pass
	InsertPerfMarker(cmdBuf, "Compute Shader: Cell Scan", color[(count++) % 3]);
	//vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScanCellPipeline);
	//vkCmdDispatch(cmdBuf, (cellSize + (1024 - 1)) / 1024, 1, 1);
	this->cellScan(cmdBuf, frames);
	EndPerfMarker(cmdBuf);

	//InsertPerfMarker(cmdBuf, "Compute Shader: Cell Scan Validate", color[(count++) % 3]);
	//vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScanCellValidationPipeline);
	//vkCmdDispatch(cmdBuf, (cellSize + (1024 - 1)) / 1024, 1, 1);
	//EndPerfMarker(cmdBuf);

	// --------------------------------------------
	// Build Hash Grid Pass
	InsertPerfMarker(cmdBuf, "Compute Shader: Build Hash Grid", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildHashGridPipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);

	// Insert barrier for Build Hash Grid Pass
	VkBufferMemoryBarrier buildHashGridPassBarriers[1] = {};

	// Barrier for m_CellStorage
	buildHashGridPassBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	buildHashGridPassBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	buildHashGridPassBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	buildHashGridPassBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buildHashGridPassBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buildHashGridPassBarriers[0].buffer = m_CellStorage[(frames + 1) % 2].buffer;
	buildHashGridPassBarriers[0].offset = 0;
	buildHashGridPassBarriers[0].size = elementCount * sizeof(uint);

	// Apply the pipeline barrier
	vkCmdPipelineBarrier(
		cmdBuf,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                  // Source stage (Fill Buffer)
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,            // Destination stage (Compute Shader)
		0,                                              // No flags
		0, nullptr,                                     // No memory barriers
		1, buildHashGridPassBarriers,				    // Buffer barriers
		0, nullptr                                      // No image barriers
	);

	// --------------------------------------------
	// Spatial Temporal Resample Pass
	InsertPerfMarker(cmdBuf, "Compute Shader: Spatial Temporal Resample", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_STResamplePipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);

	// Insert barrier for Spatial Temporal Resample Pass
	VkBufferMemoryBarrier spatialTemporalResamplePassBarriers[1] = {};

	// Barrier for m_currentReservoirs
	spatialTemporalResamplePassBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	spatialTemporalResamplePassBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	spatialTemporalResamplePassBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	spatialTemporalResamplePassBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	spatialTemporalResamplePassBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	spatialTemporalResamplePassBarriers[0].buffer = m_Reservoirs[(frames + 1) % 2].buffer;
	spatialTemporalResamplePassBarriers[0].offset = 0;
	spatialTemporalResamplePassBarriers[0].size = reseviorCount * sizeof(Reservoir);

	// Apply the pipeline barrier
	vkCmdPipelineBarrier(
		cmdBuf,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                  // Source stage (Fill Buffer)
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,            // Destination stage (Compute Shader)
		0,                                              // No flags
		0, nullptr,                                     // No memory barriers
		1, spatialTemporalResamplePassBarriers,				    // Buffer barriers
		0, nullptr                                      // No image barriers
	);

	// --------------------------------------------
	// Final Sample Pass
	InsertPerfMarker(cmdBuf, "Compute Shader: Final Sample", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_FinalSamplePipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);

	// Insert barrier for Final Sample Pass
	VkBufferMemoryBarrier finalSamplePassBarriers[1] = {};

	// Barrier for m_FinalSample
	finalSamplePassBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	finalSamplePassBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	finalSamplePassBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	finalSamplePassBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	finalSamplePassBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	finalSamplePassBarriers[0].buffer = m_FinalSample.buffer;
	finalSamplePassBarriers[0].offset = 0;
	finalSamplePassBarriers[0].size = elementCount * sizeof(FinalSample);

	// Apply the pipeline barrier
	vkCmdPipelineBarrier(
		cmdBuf,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,                  // Source stage (Fill Buffer)
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,            // Destination stage (Compute Shader)
		0,                                              // No flags
		0, nullptr,                                     // No memory barriers
		1, finalSamplePassBarriers,				    // Buffer barriers
		0, nullptr                                      // No image barriers
	);

	// --------------------------------------------
	// Final Shading Pass
	InsertPerfMarker(cmdBuf, "Compute Shader: Final Shading", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_FinalShadingPipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);
}