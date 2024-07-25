#pragma once

#include "descriptors.hpp"

#include <deque>
#include <functional>
#include <vk_mem_alloc.h>


struct AllocatedImage {
	VkImage image;
	VkImageView image_view;
	VmaAllocation allocation;
	VkExtent3D image_extent;
	VkFormat image_format;
};

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

constexpr unsigned int FRAME_OVERLAP = 2;

struct FrameData {
	VkCommandPool _command_pool;
	VkCommandBuffer _main_command_buffer;

	VkSemaphore _swapchain_semaphore, _render_semaphore;
	VkFence _render_fence;

	DeletionQueue _deletion_queue;
};

class Engine
{
public:
	Engine();
	~Engine();

	void run();

private:
	const char* TITLE = "Voxel";

	VkExtent2D _window_extent{ 1200 , 600 };
	struct SDL_Window* _window{ nullptr };
	
	bool _is_initialized{ false };
	bool _stop_rendering{ false };
	int _frame_number{ 0 };

	DeletionQueue _main_deletion_queue;

	VmaAllocator _allocator;
	
	// Initialization
	VkInstance _instance; // Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
	VkPhysicalDevice _physical_device; // GPU chosen as the default device
	VkDevice _device; // Vulkan logical device for commands
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
	AllocatedImage _draw_image;
	VkExtent2D _draw_extent;

	// Descriptors
	DescriptorAllocator global_descriptor_allocator;

	VkDescriptorSet _draw_image_descriptors;
	VkDescriptorSetLayout _draw_image_descriptor_layout;

	// Pipelines
	VkPipeline _grid_pipeline;
	VkPipelineLayout _grid_pipeline_layout;

	// Immediate Submit Structures
	VkFence _imm_fence;
	VkCommandBuffer _imm_command_buffer;
	VkCommandPool _imm_command_pool;

	// Initialization functions
	void init_vulkan();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void init_swapchain();

	void init_command_pool();

	void init_sync_structures();

	void init_descriptors();

	void init_pipelines();
	void init_grid_pipeline();
	void init_example_compute_pipelines();

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	// Render functions
	void render();
	void render_background(VkCommandBuffer cmd);
	void render_geometry(VkCommandBuffer cmd);
};


