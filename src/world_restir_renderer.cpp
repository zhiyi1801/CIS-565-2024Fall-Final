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
#include <OpenImageDenoise/oidn.hpp>

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

void* WorldRestirRenderer::mapBuffer(const nvvk::Buffer& buffer)
{
	void* data = nullptr;

	//VkResult result = m_pAlloc->map(buffer, &data);
	//if (result != VK_SUCCESS)
	{
		std::cerr << "Failed to map buffer!" << std::endl;
		return nullptr;
	}
	return data;
}

void WorldRestirRenderer::unmapBuffer(const nvvk::Buffer& buffer)
{
	m_pAlloc->unmap(buffer);
}

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

#if USE_OIDN
	oidn_device = oidn::newDevice(oidn::DeviceType::CPU);
	oidn_device.commit();

	oidn_filter = oidn_device.newFilter("RT");
	oidn_filter.set("hdr", true);  // If using HDR
	oidn_filter.set("cleanAux", true);
	oidn_filter.set("quality", OIDN_QUALITY_FAST);
	//oidn_filter.set("maxMemoryMB", 3000);

	const char* errorMessage = nullptr;
	oidn::Error error = oidn_device.getError(errorMessage);
	if (error != oidn::Error::None)
	{
		std::cerr << "OIDN create Error: " << errorMessage << std::endl;
	}
	std::cout << "OIDN initialized successfully!" << std::endl;

#endif
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

	//// Initialize FSR 3.1 API
	//ffxCreateContextDescUpscale createFsr{};
	//createFsr.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
	//createFsr.header.pNext = NULL;
	//createFsr.maxRenderSize.width = size.width;
	//createFsr.maxRenderSize.height = size.height;
	//createFsr.maxUpscaleSize.width = 960.f;   
	//createFsr.maxUpscaleSize.height = 540.f; 
	//createFsr.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;

	//ffxCreateBackendVKDesc backendDesc = {};
	//backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	//backendDesc.header.pNext = NULL;
	//backendDesc.vkDevice = m_device; 
	////backendDesc.vkPhysicalDevice = m_physicalDevice; 
	//backendDesc.vkDeviceProcAddr = vkGetDeviceProcAddr;

	//createFsr.header.pNext = &backendDesc.header;

	//ffxReturnCode_t retCode = ffxCreateContext(&m_UpscalingContext, &createFsr.header, NULL);
	//if (retCode != FFX_API_RETURN_OK) {
	//	printf("Failed to create FFX context. Error code: %d\n", retCode);
	//}

	//timer.print();
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
	m_pAlloc->destroy(m_DebugImage);
	m_pAlloc->destroy(m_DebugUintImage);
	m_pAlloc->destroy(m_IndexTempBuffer);
	m_pAlloc->destroy(m_DebugUintBuffer);
	m_pAlloc->destroy(m_DebugFloatBuffer);

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
#if USE_OIDN
	m_pAlloc->destroy(m_albedoImage);
	m_pAlloc->destroy(m_normalImage);
#endif
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

	vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
	m_pipelineLayout = VK_NULL_HANDLE;
	m_pipeline = VK_NULL_HANDLE;
	m_GbufferPipeline = VK_NULL_HANDLE;
	m_InitialSamplePipeline = VK_NULL_HANDLE;
	m_BuildHashGridPipeline = VK_NULL_HANDLE;

	//ffxDestroyContext(&m_UpscalingContext, NULL);
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

	for (int i = 0; i < 2; ++i)
	{
		m_Reservoirs[i] = m_pAlloc->createBuffer(reseviorCount * sizeof(Reservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_CellStorage[i] = m_pAlloc->createBuffer(elementCount * sizeof(uint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		m_IndexBuffer[i] = m_pAlloc->createBuffer(hashBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT| VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
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

		m_gbuffer[0] = m_pAlloc->createTexture(gbimage1, ivInfo1);
		m_gbuffer[1] = m_pAlloc->createTexture(gbimage2, ivInfo2);
		m_gbuffer[0].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_gbuffer[1].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		m_DebugImage = m_pAlloc->createTexture(DebugImage, ivInfoDebug);
		m_DebugImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		m_DebugUintImage = m_pAlloc->createTexture(DebugUintImage, ivInfoUintDebug);
		m_DebugUintImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

#if USE_OIDN
		auto albedoImageCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, VK_FORMAT_R32G32B32A32_SFLOAT,  // Float format
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, true);

		nvvk::Image albedoImage = m_pAlloc->createImage(albedoImageCreateInfo);
		NAME_VK(albedoImage.image);
		VkImageViewCreateInfo albedoIVInfo = nvvk::makeImageViewCreateInfo(albedoImage.image, albedoImageCreateInfo);
		m_albedoImage = m_pAlloc->createTexture(albedoImage, albedoIVInfo);
		m_albedoImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		auto normalImageCreateInfo = nvvk::makeImage2DCreateInfo(
			m_size, VK_FORMAT_R32G32B32A32_SFLOAT,  // Float format
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, true);

		nvvk::Image normalImage = m_pAlloc->createImage(normalImageCreateInfo);
		NAME_VK(normalImage.image);
		VkImageViewCreateInfo normalIVInfo = nvvk::makeImageViewCreateInfo(normalImage.image, normalImageCreateInfo);
		m_normalImage = m_pAlloc->createTexture(normalImage, normalIVInfo);
		m_normalImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
#endif
	}

	// Setting the image layout for both color and depth
	{
		nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
		auto              cmdBuf = genCmdBuf.createCommandBuffer();
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_gbuffer[1].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_DebugImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_DebugUintImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
#if USE_OIDN
		//  Transform the layout of albedo and normal images
		nvvk::cmdBarrierImageLayout(cmdBuf, m_albedoImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		nvvk::cmdBarrierImageLayout(cmdBuf, m_normalImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
#endif
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
	m_bind.addBinding({ ReSTIRBindings::eIndexTemp, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });

	// Debug images
	m_bind.addBinding({ ReSTIRBindings::eDebugUintImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eDebugImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	// Debug buffers
	m_bind.addBinding({ ReSTIRBindings::eDebugUintBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eDebugFloatBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flag });
#if USE_OIDN 1
	m_bind.addBinding({ ReSTIRBindings::eAlbedoImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
	m_bind.addBinding({ ReSTIRBindings::eNormalImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, flag });
#endif

	m_descPool = m_bind.createPool(m_device, m_descSet.size(), VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
	CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
	CREATE_NAMED_VK(m_descSet[0], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));
	CREATE_NAMED_VK(m_descSet[1], nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

	updateDescriptorSet();
}

void WorldRestirRenderer::updateDescriptorSet()
{
#if USE_OIDN
	std::array<VkWriteDescriptorSet, 19> writes;
#else
	std::array<VkWriteDescriptorSet, 17> writes;
#endif
	VkDeviceSize fullScreenSize = m_size.width * m_size.height;
	VkDeviceSize elementCount = fullScreenSize;
	VkDeviceSize hashBufferSize = m_CellSize * sizeof(uint32_t);
	VkDeviceSize reseviorCount = 2 * elementCount;

	for (int i = 0; i < 2; i++) {
		VkDescriptorBufferInfo initialResvervoirBufInfo = { m_Reservoirs[i].buffer, 0, elementCount * sizeof(Reservoir) };
		VkDescriptorBufferInfo reservoirBufInfo = { m_Reservoirs[i].buffer, 0, reseviorCount * sizeof(Reservoir)};
		VkDescriptorBufferInfo appendBufInfo = { m_AppendBuffer.buffer, 0, elementCount * sizeof(HashAppendData) };
		VkDescriptorBufferInfo finalSampleBufInfo = { m_FinalSample.buffer, 0, elementCount * sizeof(FinalSample) };
		VkDescriptorBufferInfo initialSampleBufInfo = { m_InitialSamples.buffer, 0, elementCount * sizeof(InitialSample) };
		VkDescriptorBufferInfo reconnectionBufInfo = { m_ReconnectionData.buffer, 0, elementCount * sizeof(ReconnectionData) };

		VkDescriptorBufferInfo cellStorageBufInfo = { m_CellStorage[i].buffer, 0, elementCount * sizeof(uint) };
		VkDescriptorBufferInfo indexBufInfo = { m_IndexBuffer[i].buffer, 0, hashBufferSize};
		VkDescriptorBufferInfo indexTempBufInfo = { m_IndexTempBuffer.buffer, 0, hashBufferSize};
		VkDescriptorBufferInfo checkSumBufInfo = { m_CheckSumBuffer[i].buffer, 0, hashBufferSize };
		VkDescriptorBufferInfo cellCounterBufInfo = { m_CellCounter[i].buffer, 0, hashBufferSize };

		writes[0] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eLastGbuffer, &m_gbuffer[i].descriptor);
		writes[1] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCurrentGbuffer, &m_gbuffer[!i].descriptor);

		writes[2] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eInitialReservoirs, &initialResvervoirBufInfo);
		writes[3] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eReservoirs, &reservoirBufInfo);
		writes[4] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eAppend, &appendBufInfo);
		writes[5] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eFinal, &finalSampleBufInfo);
		writes[6] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCell, &cellStorageBufInfo);
		writes[7] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eIndex, &indexBufInfo);
		writes[8] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCheckSum, &checkSumBufInfo);
		writes[9] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eCellCounter, &cellCounterBufInfo);
		writes[10] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eInitialSamples, &initialSampleBufInfo);
		writes[11] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eReconnection, &reconnectionBufInfo);
		writes[12] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eIndexTemp, &indexTempBufInfo);

		// debug images desc
		writes[13] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eDebugUintImage, &m_DebugUintImage.descriptor);
		writes[14] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eDebugImage, &m_DebugImage.descriptor);

		// debug buffers desc
		VkDescriptorBufferInfo debugUintBufInfo = { m_DebugUintBuffer.buffer, 0, m_DebugBufferSize * sizeof(uint32_t) };
		VkDescriptorBufferInfo debugFloatBufInfo = { m_DebugFloatBuffer.buffer, 0, m_DebugBufferSize * sizeof(float) };
		writes[15] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eDebugUintBuffer, &debugUintBufInfo);
		writes[16] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eDebugFloatBuffer, &debugFloatBufInfo);
#if USE_OIDN
		writes[17] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eAlbedoImage, &m_albedoImage.descriptor);
		writes[18] = m_bind.makeWrite(m_descSet[i], ReSTIRBindings::eNormalImage, &m_normalImage.descriptor);
#endif
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

		// ���� Pipeline Barrier ȷ����һ��д�������
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
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // Դ�׶�
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,  // Ŀ��׶�
			0,                                    // �������־
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
	vkCmdFillBuffer(cmdBuf, m_DebugUintBuffer.buffer, 0, m_DebugBufferSize * sizeof(uint), 0);
	vkCmdFillBuffer(cmdBuf, m_DebugFloatBuffer.buffer, 0, m_DebugBufferSize * sizeof(float), 0);

	// Insert a barrier
	VkBufferMemoryBarrier bufferBarriers[4] = {};

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
	//vkCmdDispatch(cmdBuf, (cellSize + (1024 - 1)) / 1024, 1, 1);
	this->cellScan(cmdBuf, frames);
	EndPerfMarker(cmdBuf);

	//InsertPerfMarker(cmdBuf, "Compute Shader: Cell Scan Validate", color[(count++) % 3]);
	//vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScanCellValidationPipeline);
	//vkCmdDispatch(cmdBuf, (cellSize + (1024 - 1)) / 1024, 1, 1);
	//EndPerfMarker(cmdBuf);

	InsertPerfMarker(cmdBuf, "Compute Shader: Build Hash Grid", color[(count++) % 3]);
	vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildHashGridPipeline);
	vkCmdDispatch(cmdBuf, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
	EndPerfMarker(cmdBuf);



#if USE_OIDN
	InsertPerfMarker(cmdBuf, "Open Image Denoiser", color[(count++) % 3]);
	runOIDN(cmdBuf, size, profiler, descSets, frames);
	EndPerfMarker(cmdBuf);
#endif // USE_OIDN
	
}

void WorldRestirRenderer::runOIDN(const VkCommandBuffer& cmdBuf, const VkExtent2D& size, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet>& descSets, uint frames) {
	
	// 1. Define the required size and flags
	VkDeviceSize imageSize = size.width * size.height * 4 * sizeof(float); // Assume 4 floats per pixel (RGBA32F)

	VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags memoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	// 2. Create a CPU accessible buffer
	nvvk::Buffer colorBuffer = m_pAlloc->createBuffer(imageSize, usageFlags, memoryPropertyFlags);
	nvvk::Buffer albedoBuffer = m_pAlloc->createBuffer(imageSize, usageFlags, memoryPropertyFlags);
	nvvk::Buffer normalBuffer = m_pAlloc->createBuffer(imageSize, usageFlags, memoryPropertyFlags);

	// 3. Create a command buffer
	nvvk::CommandPool cmdPool(m_device, m_queueIndex);
	VkCommandBuffer copyCmdBuf = cmdPool.createCommandBuffer();

	// 4. Change the image layout to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
	nvvk::cmdBarrierImageLayout(copyCmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	nvvk::cmdBarrierImageLayout(copyCmdBuf, m_albedoImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	nvvk::cmdBarrierImageLayout(copyCmdBuf, m_normalImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

	// 5. Copy from image to buffer
	VkBufferImageCopy region = {};
	region.bufferOffset = 0;
	region.bufferRowLength = 0; // Tightly packed
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = { size.width, size.height, 1 };

	vkCmdCopyImageToBuffer(copyCmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, colorBuffer.buffer, 1, &region);
	vkCmdCopyImageToBuffer(copyCmdBuf, m_albedoImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, albedoBuffer.buffer, 1, &region);
	vkCmdCopyImageToBuffer(copyCmdBuf, m_normalImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, normalBuffer.buffer, 1, &region);

	// 6. Change the image layout back to VK_IMAGE_LAYOUT_GENERAL
	nvvk::cmdBarrierImageLayout(copyCmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
	nvvk::cmdBarrierImageLayout(copyCmdBuf, m_albedoImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
	nvvk::cmdBarrierImageLayout(copyCmdBuf, m_normalImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

	// 7. Submit and wait for the copy command
	cmdPool.submitAndWait(copyCmdBuf);

	// 8. Map the buffer and get the pointer
	void* colorData = m_pAlloc->map(colorBuffer);
	void* albedoData = m_pAlloc->map(albedoBuffer);
	void* normalData = m_pAlloc->map(normalBuffer);



	// 9. Setup and execute the OIDN filter
	oidn_filter.setImage("color", colorData, oidn::Format::Float3, size.width, size.height);
	oidn_filter.setImage("albedo", albedoData, oidn::Format::Float3, size.width, size.height);
	oidn_filter.setImage("normal", normalData, oidn::Format::Float3, size.width, size.height);
	oidn_filter.setImage("output", colorData, oidn::Format::Float3, size.width, size.height); 

	oidn_filter.commit();

	auto start = std::chrono::high_resolution_clock::now();
	oidn_filter.execute();

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Task duration: " << elapsed.count() << " seconds" << std::endl;

	// 10. Check for errors
	const char* errorMessage = nullptr;
	oidn::Error error = oidn_device.getError(errorMessage);
	if (error != oidn::Error::None)
	{
		std::cerr << "OIDN Running Error: " << errorMessage << std::endl;
	}

	// 11. Unmap the buffer
	m_pAlloc->unmap(colorBuffer);
	m_pAlloc->unmap(albedoBuffer);
	m_pAlloc->unmap(normalBuffer);

	// 12. Copy the processed data back to the GPU image
	VkCommandBuffer copyBackCmdBuf = cmdPool.createCommandBuffer();

	// Change the image layout to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
	nvvk::cmdBarrierImageLayout(copyBackCmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	// Copy from buffer to image
	vkCmdCopyBufferToImage(copyBackCmdBuf, colorBuffer.buffer, m_gbuffer[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	//  Change the image layout back to VK_IMAGE_LAYOUT_GENERAL
	nvvk::cmdBarrierImageLayout(copyBackCmdBuf, m_gbuffer[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

	cmdPool.submitAndWait(copyBackCmdBuf);

	// 13. Destroy the buffer
	m_pAlloc->destroy(colorBuffer);
	m_pAlloc->destroy(albedoBuffer);
	m_pAlloc->destroy(normalBuffer);
}