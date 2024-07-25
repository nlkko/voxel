#include "engine.hpp"

#include "types.hpp"
#include "images.hpp"
#include "initializers.hpp"
#include "descriptors.hpp"
#include "pipelines.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <VkBootstrap.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

// dear imgui
//#include <imgui.h>
//#include <backends/imgui_impl_sdl2.h>
//#include <backends/imgui_impl_vulkan.h>

#include<thread>
#include <chrono>


//TODO:
// * Use pure vulkan instead of bootstrap
// * Use several queue families
//      - 1 for graphics, 1 for UI
// * Use several command pools
// * DeleteQueue via vulkan handles

constexpr bool use_validation_layers_ = true;

Engine::Engine()
{
    fmt::print("\033[1;38;5;208m[ Initializing... ]\033[0m\n");

    // We initialize SDL and create a window with it. 
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow(
        TITLE,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _window_extent.width,
        _window_extent.height,
        window_flags
    );

    init_vulkan();

    init_swapchain();

    init_command_pool();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    _is_initialized = true;
}

Engine::~Engine()
{
    fmt::print("\033[1;38;5;208m[ Cleaning up... ]\033[0m\n");

    if (_is_initialized) {

        //make sure the gpu has stopped doing its things
        vkDeviceWaitIdle(_device);

        _main_deletion_queue.flush();

        for (int i = 0; i < FRAME_OVERLAP; i++) {

            vkDestroyCommandPool(_device, _frames[i]._command_pool, nullptr);

            //destroy sync objects
            vkDestroyFence(_device, _frames[i]._render_fence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._render_semaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchain_semaphore, nullptr);
        }

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);

        vkDestroyDevice(_device, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

}

void Engine::init_vulkan()
{
    // initializing instance & debug
    vkb::InstanceBuilder builder;

    auto instance_return = builder.set_app_name(TITLE)
        .request_validation_layers(use_validation_layers_)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkb_instance = instance_return.value();

    _instance = vkb_instance.instance;
    _debug_messenger = vkb_instance.debug_messenger;

    // creating window surface
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    // vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features_13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features_13.dynamicRendering = true;
    features_13.synchronization2 = true;

    // vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features_12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features_12.bufferDeviceAddress = true;
    features_12.descriptorIndexing = true;

    // select GPU with vkb
    vkb::PhysicalDeviceSelector selector{ vkb_instance };
    vkb::PhysicalDevice physical_device = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features_13)
        .set_required_features_12(features_12)
        .set_surface(_surface)
        .select()
        .value();

    // create device & physical device
    vkb::DeviceBuilder device_builder{ physical_device };

    vkb::Device vkb_device = device_builder.build().value();

    _device = vkb_device.device;
    _physical_device = physical_device.physical_device;

    // use vkb to get graphics queue
    _graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    _graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = _physical_device;
    allocator_info.device = _device;
    allocator_info.instance = _instance;
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocator_info, &_allocator);

    _main_deletion_queue.push_function([&]() {
        vmaDestroyAllocator(_allocator);
        });
}

void Engine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchain_builder{ _physical_device, _device, _surface };

    _swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkb_swapchain = swapchain_builder
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchain_image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchain_extent = vkb_swapchain.extent;
    
    //store swapchain and its related images
    _swapchain = vkb_swapchain.swapchain;
    _swapchain_images = vkb_swapchain.get_images().value();
    _swapchain_image_views = vkb_swapchain.get_image_views().value();
}

void Engine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    // destroy swapchain resources
    for (int i = 0; i < _swapchain_image_views.size(); i++) {

        vkDestroyImageView(_device, _swapchain_image_views[i], nullptr);

    }
}

void Engine::init_swapchain()
{
    create_swapchain(_window_extent.width, _window_extent.height);

    // match extent with window extent
    VkExtent3D drawImageExtent = {
        _window_extent.width,
        _window_extent.height,
        1
    };

    //hardcoding the draw format to 32 bit float
    _draw_image.image_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    _draw_image.image_extent = drawImageExtent;

    VkImageUsageFlags draw_image_usages{};
    draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_STORAGE_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_draw_image.image_format, draw_image_usages, drawImageExtent);

    // allocate it from local gpu memory
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    //allocate and create the image
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_draw_image.image, &_draw_image.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_draw_image.image_format, _draw_image.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_draw_image.image_view));

    //add to deletion queues
    _main_deletion_queue.push_function([=]() {
        vkDestroyImageView(_device, _draw_image.image_view, nullptr);
        vmaDestroyImage(_allocator, _draw_image.image, _draw_image.allocation);
        });
}

void Engine::init_command_pool()
{
    //create a command pool for commands submitted to the graphics queue.
    //we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo command_pool_info = vkinit::command_pool_create_info(_graphics_queue_family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; i++) {

        VK_CHECK(vkCreateCommandPool(_device, &command_pool_info, nullptr, &_frames[i]._command_pool));

        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmd_alloc_info = {};
        cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc_info.pNext = nullptr;
        cmd_alloc_info.commandPool = _frames[i]._command_pool;
        cmd_alloc_info.commandBufferCount = 1;
        cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmd_alloc_info, &_frames[i]._main_command_buffer));
    }

    // command pool for immediate submits<
    VK_CHECK(vkCreateCommandPool(_device, &command_pool_info, nullptr, &_imm_command_pool));

    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_imm_command_pool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_imm_command_buffer));

    _main_deletion_queue.push_function([=]() {
        vkDestroyCommandPool(_device, _imm_command_pool, nullptr);
        });
}

void Engine::init_sync_structures()
{
    // create syncronization structures

    // 1 Fence to control when GPU has finished rendering frame.
    // 2 Semapphores to synchronize with swapchain

    // Fence starts signalled so we can wait on the first frame

    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._render_fence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchain_semaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._render_semaphore));
    }

    // immediate submit fence
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_imm_fence));
    _main_deletion_queue.push_function([=]() { vkDestroyFence(_device, _imm_fence, nullptr); });
}

void Engine::init_descriptors()
{
    // create descriptor pool that will hold 10 sets with 1 image each
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 }
    };

    global_descriptor_allocator.init_pool(_device, 10, sizes);

    // create descriptor set layout for compute draw
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _draw_image_descriptor_layout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // allocate a descriptor set for draw image
    _draw_image_descriptors = global_descriptor_allocator.allocate(_device, _draw_image_descriptor_layout);

    VkDescriptorImageInfo img_info{};
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    img_info.imageView = _draw_image.image_view;

    VkWriteDescriptorSet draw_image_write = {};
    draw_image_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    draw_image_write.pNext = nullptr;

    draw_image_write.dstBinding = 0;
    draw_image_write.dstSet = _draw_image_descriptors;
    draw_image_write.descriptorCount = 1;
    draw_image_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    draw_image_write.pImageInfo = &img_info;

    // update the descriptor set with the image info
    vkUpdateDescriptorSets(_device, 1, &draw_image_write, 0, nullptr);

    // clean up descriptor allocator and new layout
    _main_deletion_queue.push_function([&]() {
        global_descriptor_allocator.destroy_pool(_device);
        vkDestroyDescriptorSetLayout(_device, _draw_image_descriptor_layout, nullptr);
        });
}

void Engine::init_pipelines()
{
    //init_example_compute_pipelines();
    init_grid_pipeline();
}

void Engine::init_grid_pipeline()
{
    VkShaderModule grid_frag_shader;
    if (!vkutil::load_shader_module("../../shaders/grid.frag.spv", _device, &grid_frag_shader)) {
        fmt::print("Error when building the grid fragment shader module\n");
    }
    else {
        fmt::print("Grid fragment shader succesfully loaded\n");
    }

    VkShaderModule grid_vert_shader;
    if (!vkutil::load_shader_module("../../shaders/grid.vert.spv", _device, &grid_vert_shader)) {
        fmt::print("Error when building the grid vertex shader module\n");
    }
    else {
        fmt::print("Grid vertex shader succesfully loaded\n");
    }

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_grid_pipeline_layout));
}

void Engine::init_example_compute_pipelines()
{
    // create layour
    VkPipelineLayoutCreateInfo compute_layout{};
    compute_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    compute_layout.pNext = nullptr;
    compute_layout.pSetLayouts = &_draw_image_descriptor_layout;
    compute_layout.setLayoutCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &compute_layout, nullptr, &_grid_pipeline_layout));

    // shader
    VkShaderModule compute_draw_shader;

    if (!vkutil::load_shader_module("../../shaders/gradient.comp.spv", _device, &compute_draw_shader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.pNext = nullptr;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = compute_draw_shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo compute_pipeline_create_info{};
    compute_pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipeline_create_info.pNext = nullptr;
    compute_pipeline_create_info.layout = _grid_pipeline_layout;
    compute_pipeline_create_info.stage = stage_info;

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr, &_grid_pipeline));


    // clean up
    vkDestroyShaderModule(_device, compute_draw_shader, nullptr);

    _main_deletion_queue.push_function([&]() {
        vkDestroyPipelineLayout(_device, _grid_pipeline_layout, nullptr);
        vkDestroyPipeline(_device, _grid_pipeline, nullptr);
        });
}

void Engine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_imm_fence));
    VK_CHECK(vkResetCommandBuffer(_imm_command_buffer, 0));

    VkCommandBuffer cmd = _imm_command_buffer;

    VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphics_queue, 1, &submit, _imm_fence));

    VK_CHECK(vkWaitForFences(_device, 1, &_imm_fence, true, 9999999999));
}

void Engine::render()
{
    // wait until the gpu has finished rendering the last frame. Timeout of 1 second
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._render_fence, true, 1000000000));

    get_current_frame()._deletion_queue.flush();

    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._render_fence));

    // request image from the swapchain
    uint32_t swapchain_image_index;
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchain_semaphore, nullptr, &swapchain_image_index));

    // reset command buffer
    VkCommandBuffer cmd = get_current_frame()._main_command_buffer;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    _draw_extent.width = _draw_image.image_extent.width;
    _draw_extent.height = _draw_image.image_extent.height;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));


    // transition swapchain image format
    vkutil::transition_image(cmd, _draw_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    render_scene(cmd);

    // transition the draw image and the swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, _draw_image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // copy draw image to swapchain image
    vkutil::copy_image_to_image(cmd, _draw_image.image, _swapchain_images[swapchain_image_index], _draw_extent, _swapchain_extent);

    // transition swapchain image format to present format
    vkutil::transition_image(cmd, _swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // submit to queue
    VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo wait_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchain_semaphore);
    VkSemaphoreSubmitInfo signal_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._render_semaphore);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, &signal_info, &wait_info);

    VK_CHECK(vkQueueSubmit2(_graphics_queue, 1, &submit, get_current_frame()._render_fence));

    //prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the _renderSemaphore for that, 
    // as its necessary that drawing commands have finished before the image is displayed to the user
    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pNext = nullptr;
    present_info.pSwapchains = &_swapchain;
    present_info.swapchainCount = 1;

    present_info.pWaitSemaphores = &get_current_frame()._render_semaphore;
    present_info.waitSemaphoreCount = 1;

    present_info.pImageIndices = &swapchain_image_index;

    VK_CHECK(vkQueuePresentKHR(_graphics_queue, &present_info));

    _frame_number++;
}

void Engine::render_scene(VkCommandBuffer cmd)
{
    // make a clear-color from frame number. This will flash with a 120 frame period.
    /*
    VkClearColorValue clear_value = { 0.082f, 0.184f, 0.31f, 1.0f };

    VkImageSubresourceRange clear_range = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);

    vkCmdClearColorImage(cmd, _draw_image.image, VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1, &clear_range);
    */

    // bind the gradient drawing compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _grid_pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _grid_pipeline_layout, 0, 1, &_draw_image_descriptors, 0, nullptr);

    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(_draw_extent.width / 16.0), std::ceil(_draw_extent.height / 16.0), 1);
}

void Engine::run()
{
    fmt::print("\033[1;38;5;208m[ Running... ]\033[0m\n");

    SDL_Event e;
    bool quit = false;

    // main loop
    while (!quit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                quit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    _stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    _stop_rendering = false;
                }
            }

            // Escape using ESC key
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) {
                    quit = true;
                }
            }
        }

        // Do not render if minimized
        if (_stop_rendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        else {
            render();
        }

    }
} 