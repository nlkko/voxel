#pragma once

#include "descriptors.hpp"
#include "loader.hpp"

#include <deque>
#include <functional>
#include <vk_mem_alloc.h>


struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call functors
		}

		deletors.clear();
	}
};

struct FrameData {
	VkCommandPool _command_pool;
	VkCommandBuffer _main_command_buffer;

	VkSemaphore _swapchain_semaphore, _render_semaphore;
	VkFence _render_fence;

	DeletionQueue _deletion_queue;
	DescriptorAllocatorGrowable _frame_descriptors;
};

struct GLTFMetallic_Roughness {
	MaterialPipeline opaque_pipeline;
	MaterialPipeline transparent_pipeline;

	VkDescriptorSetLayout material_layout;

	struct MaterialConstants {
		glm::vec4 color_factors;
		glm::vec4 metal_rough_factors;
		//padding, we need it anyway for uniform buffers
		glm::vec4 extra[14];
	};

	struct MaterialResources {
		AllocatedImage color_image;
		VkSampler color_sampler;
		AllocatedImage metal_rough_image;
		VkSampler metal_rough_sampler;
		VkBuffer data_buffer;
		uint32_t data_buffer_offset;
	};

	DescriptorWriter writer;

	void build_pipelines(Engine* engine);
	void clear_resources(VkDevice device);

	MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct AllocatedImage {
	VkImage image;
	VkImageView image_view;
	VmaAllocation allocation;
	VkExtent3D image_extent;
	VkFormat image_format;
};

struct RenderObject {
	uint32_t index_count;
	uint32_t first_index;
	VkBuffer index_buffer;

	MaterialInstance* material;

	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

class Engine
{
public:
	Engine();
	~Engine();

	// Initialization
	VkDevice _device; // Vulkan logical device for commands

	// Descriptors
	VkDescriptorSetLayout _gpu_scene_data_descriptor_layout;

	// Draw Resources
	AllocatedImage _draw_image;
	AllocatedImage _depth_image;

	GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	void run();

private:
	const char* TITLE = "Voxel";

	VkExtent2D _window_extent{ 1200 , 600 };
	struct SDL_Window* _window{ nullptr };
	
	bool _is_initialized{ false };
	bool _stop_rendering{ false };
	bool _resize_requested{ false };
	int _frame_number{ 0 };

	DeletionQueue _main_deletion_queue;

	VmaAllocator _allocator;
	
	// Initialization
	VkInstance _instance; // Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
	VkPhysicalDevice _physical_device; // GPU chosen as the default device
	
	VkSurfaceKHR _surface; // Vulkan window surface

	// Swapchain
	VkSwapchainKHR _swapchain;
	VkFormat _swapchain_image_format;

	std::vector<VkImage> _swapchain_images;
	std::vector<VkImageView> _swapchain_image_views;
	VkExtent2D _swapchain_extent;

	// Command Pool
	FrameData _frames[FRAME_OVERLAP];
	FrameData& get_current_frame() { return _frames[_frame_number % FRAME_OVERLAP]; };

	VkQueue _graphics_queue;
	uint32_t _graphics_queue_family;

	// Draw Resources
	VkExtent2D _draw_extent;

	GPUSceneData scene_data;
	

	// Descriptors
	DescriptorAllocator global_descriptor_allocator;

	VkDescriptorSet _draw_image_descriptors;
	VkDescriptorSetLayout _draw_image_descriptor_layout;

	// Pipelines
	VkPipeline _grid_pipeline;
	VkPipelineLayout _grid_pipeline_layout;
	
	VkPipeline _voxel_pipeline;
	VkPipelineLayout _voxel_pipeline_layout;

	// Immediate Submit Structures
	VkFence _imm_fence;
	VkCommandBuffer _imm_command_buffer;
	VkCommandPool _imm_command_pool;

	// Buffer
	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

	void destroy_buffer(const AllocatedBuffer& buffer);

	// Initialization functions
	void init_vulkan();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void resize_swapchain();
	void init_swapchain();

	void init_command_pool();

	void init_sync_objects();

	void create_sync_objects(); 
	void destroy_sync_objects();

	void init_descriptors();

	void init_pipelines();

	void init_imgui();

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	// Render functions
	void render();
	void render_background(VkCommandBuffer cmd);
	void render_geometry(VkCommandBuffer cmd);
	void render_imgui(VkCommandBuffer cmd, VkImageView target_image_view);

	// Textures
	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage& img);

	// Temp
	VkPipelineLayout _mesh_pipeline_layout;
	VkPipeline _mesh_pipeline;

	void init_mesh_pipeline();

	void init_data();

	std::vector<std::shared_ptr<MeshAsset>> test_meshes;

	AllocatedImage _error_checkerboard_image;

	VkSampler _default_sampler_linear;
	VkSampler _default_sampler_nearest;

	VkDescriptorSetLayout _single_image_descriptor_layout;


};


