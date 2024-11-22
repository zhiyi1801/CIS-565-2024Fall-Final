#pragma once

#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/profiler_vk.hpp"

#include "renderer.h"
#include "shaders/host_device.h"

using nvvk::SBTWrapper;

class WorldRestirRenderer : public Renderer
{
public:
	void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator) override;
	void destroy() override;
	void create(const VkExtent2D& size, std::vector<VkDescriptorSetLayout>& rtDescSetLayouts, Scene* _scene = nullptr) override;
	void run(const VkCommandBuffer& cmdBuf, const VkExtent2D& size, nvvk::ProfilerVK& profiler, std::vector<VkDescriptorSet>& descSets, uint frames) override;
	void update(const VkExtent2D& size);
	void createBuffer();
	void createImage();
	void createDescriptorSet();
	void updateDescriptorSet();
	const std::string name() override { return std::string("WorldRestir"); }

	void setRecompile(bool recompile) { m_Recompile = recompile; }
	void setStateChanged(bool stateChanged) { m_StateChanged = stateChanged; }

	void cellScan(const VkCommandBuffer& cmdBuf, const int frames);

private:
	nvvk::ResourceAllocator* m_pAlloc{ nullptr };  // Allocator for buffer, images, acceleration structures
	nvvk::DebugUtil          m_debug;            // Utility to name objects
	VkDevice                 m_device{ VK_NULL_HANDLE };
	uint32_t                 m_queueIndex{ 0 };

	// Depth 32bit, Normal 32bit, Metallic 8bit, Roughness 8bit, IOR 8bit, Transmission 8bit, Albedo 24bit, Hashed Material ID 8bit
	VkFormat m_gbufferFormat{ VK_FORMAT_R32G32B32A32_UINT };
	VkFormat m_DebugImageFormat{ VK_FORMAT_R32G32B32A32_SFLOAT };
	VkFormat m_DebugUintImageFormat{ VK_FORMAT_R32G32B32A32_UINT };

	std::array<nvvk::Texture, 2> m_gbuffer;
	nvvk::Texture m_DebugImage;
	nvvk::Texture m_DebugUintImage;

	nvvk::Buffer m_InitialReservoir;
	nvvk::Buffer m_AppendBuffer;
	nvvk::Buffer m_FinalSample;
	nvvk::Buffer m_InitialSamples;
	nvvk::Buffer m_ReconnectionData;
	nvvk::Buffer m_IndexTempBuffer;
	std::array<nvvk::Buffer, 2> m_Reservoirs;	/// store for both temporal and spatial reservoir
	std::array<nvvk::Buffer, 2> m_CellStorage;
	std::array<nvvk::Buffer, 2> m_IndexBuffer;
	std::array<nvvk::Buffer, 2> m_CheckSumBuffer;
	std::array<nvvk::Buffer, 2> m_CellCounter;

	// Debug buffer
	nvvk::Buffer m_DebugUintBuffer;
	nvvk::Buffer m_DebugFloatBuffer;
	
	bool m_Recompile = true;
	bool m_StateChanged = false;

	VkExtent2D m_size{};
	nvvk::DescriptorSetBindings m_bind;
	VkDescriptorPool      m_descPool{ VK_NULL_HANDLE };
	VkDescriptorSetLayout m_descSetLayout{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, 2> m_descSet{ VK_NULL_HANDLE };

	VkPipelineLayout  m_pipelineLayout{ VK_NULL_HANDLE };

	VkPipeline m_pipeline{ VK_NULL_HANDLE };
	VkPipeline m_InitialSamplePipeline{ VK_NULL_HANDLE };
	VkPipeline m_GbufferPipeline{ VK_NULL_HANDLE };
	VkPipeline m_InitialReservoirPipeline{ VK_NULL_HANDLE };
	VkPipeline m_BuildHashGridPipeline{ VK_NULL_HANDLE };
	VkPipeline m_ScanCellPipeline{ VK_NULL_HANDLE };
	VkPipeline m_ScanCellValidationPipeline{ VK_NULL_HANDLE };
	VkPipeline m_STResamplePipeline{ VK_NULL_HANDLE };
	VkPipeline m_FinalSamplePipeline{ VK_NULL_HANDLE };
	VkPipeline m_FinalShadingPipeline{ VK_NULL_HANDLE };

	VkDeviceSize m_CellSize;
	VkDeviceSize m_DebugBufferSize;
};