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
#include <array>


//TODO:
// * Use pure vulkan instead of bootstrap
// * Use several queue families
//      - 1 for graphics, 1 for UI
// * Use several command pools
// * DeleteQueue via vulkan handles
// * Create a thread for deleting/reusing buffers as well execute upload

constexpr bool use_validation_layers_ = true;

Engine::Engine()
{
    fmt::print(fg(fmt::color::orange) | fmt::emphasis::bold, "[ Initializing ]\n");

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

    init_data();

    _is_initialized = true;
}

Engine::~Engine()
{
    fmt::print(fg(fmt::color::orange) | fmt::emphasis::bold, "[ Cleaning Up ]\n");

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

void Engine::init_data()
{
    std::array<Vertex, 4> rect_vertices;

    rect_vertices[0].position = { 0.5,-0.5, 0 };
    rect_vertices[1].position = { 0.5,0.5, 0 };
    rect_vertices[2].position = { -0.5,-0.5, 0 };
    rect_vertices[3].position = { -0.5,0.5, 0 };

    rect_vertices[0].color = { 0,0, 0,1 };
    rect_vertices[1].color = { 0.5,0.5,0.5 ,1 };
    rect_vertices[2].color = { 1,0, 0,1 };
    rect_vertices[3].color = { 0,1, 0,1 };

    std::array<uint32_t, 6> rect_indices;

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    rectangle = upload_mesh(rect_indices, rect_vertices);

    //delete the rectangle data on engine shutdown
    _main_deletion_queue.push_function([&]() {
        destroy_buffer(rectangle.index_buffer);
        destroy_buffer(rectangle.vertex_buffer);
        });
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
    init_mesh_pipeline();
}

void Engine::init_mesh_pipeline()
{
    VkShaderModule grid_frag_shader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle.frag.spv", _device, &grid_frag_shader)) {
        fmt::print(fg(fmt::color::red), "Error when building the grid fragment shader module\n");
    }
    else {
        fmt::print(fg(fmt::color::green), "Fragment shader successfully loaded\n");
    }

    VkShaderModule grid_vert_shader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle_mesh.vert.spv", _device, &grid_vert_shader)) {
        fmt::print(fg(fmt::color::red), "Error when building the grid vertex shader module\n");
    }
    else {
        fmt::print(fg(fmt::color::green), "Vertex shader successfully loaded\n");
    }

    VkPushConstantRange buffer_range{};
    buffer_range.offset = 0;
    buffer_range.size = sizeof(GPUDrawPushConstants);
    buffer_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &buffer_range;
    pipeline_layout_info.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_mesh_pipeline_layout));

    PipelineBuilder pipeline_builder;

    //use the triangle layout we created
    pipeline_builder._pipeline_layout = _mesh_pipeline_layout;
    //connecting the vertex and pixel shaders to the pipeline
    pipeline_builder.set_shaders(grid_vert_shader, grid_frag_shader, "main");
    //it will draw triangles
    pipeline_builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    //filled triangles
    pipeline_builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    //no backface culling
    pipeline_builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    //no multisampling
    pipeline_builder.set_multisampling_none();
    //no blending
    pipeline_builder.disable_blending();

    pipeline_builder.disable_depthtest();

    //connect the image format we will draw into, from draw image
    pipeline_builder.set_color_attachment_format(_draw_image.image_format);
    pipeline_builder.set_depth_format(VK_FORMAT_UNDEFINED);

    //finally build the pipeline
    _mesh_pipeline = pipeline_builder.build_pipeline(_device);

    //clean structures
    vkDestroyShaderModule(_device, grid_frag_shader, nullptr);
    vkDestroyShaderModule(_device, grid_vert_shader, nullptr);

    _main_deletion_queue.push_function([&]() {
        vkDestroyPipelineLayout(_device, _mesh_pipeline_layout, nullptr);
        vkDestroyPipeline(_device, _mesh_pipeline, nullptr);
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

AllocatedBuffer Engine::create_buffer(size_t alloc_size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
{
    // allocate buffer
    VkBufferCreateInfo buffer_info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    buffer_info.pNext = nullptr;
    buffer_info.size = alloc_size;

    buffer_info.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memory_usage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer new_buffer;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &buffer_info, &vmaallocInfo, &new_buffer.buffer, &new_buffer.allocation,
        &new_buffer.info));

    return new_buffer;
}

void Engine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers Engine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertex_buffer_size = vertices.size() * sizeof(Vertex);
    const size_t index_buffer_size = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers new_surface;

    // create vertex buffer
    new_surface.vertex_buffer = create_buffer(vertex_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    // find the adress of the vertex buffer
    VkBufferDeviceAddressInfo deviceAdressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer = new_surface.vertex_buffer.buffer };
    new_surface.vertex_buffer_address = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    // create index buffer
    new_surface.index_buffer = create_buffer(index_buffer_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging = create_buffer(vertex_buffer_size + index_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.allocation->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertex_buffer_size);
    // copy index buffer
    memcpy((char*)data + vertex_buffer_size, indices.data(), index_buffer_size);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{ 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertex_buffer_size;

        vkCmdCopyBuffer(cmd, staging.buffer, new_surface.vertex_buffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{ 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertex_buffer_size;
        indexCopy.size = index_buffer_size;

        vkCmdCopyBuffer(cmd, staging.buffer, new_surface.index_buffer.buffer, 1, &indexCopy);
        });

    destroy_buffer(staging);

    return new_surface;
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

    render_background(cmd);

    vkutil::transition_image(cmd, _draw_image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    render_geometry(cmd);

    // transition the draw image and the swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, _draw_image.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
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

void Engine::render_background(VkCommandBuffer cmd)
{
    // make a clear-color from frame number. This will flash with a 120 frame period.
    VkClearColorValue clear_value = { 0.082f, 0.184f, 0.31f, 1.0f };

    VkImageSubresourceRange clear_range = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);

    vkCmdClearColorImage(cmd, _draw_image.image, VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1, &clear_range);
    

    /*
    // bind the gradient drawing compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _grid_pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _grid_pipeline_layout, 0, 1, &_draw_image_descriptors, 0, nullptr);

    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(_draw_extent.width / 16.0), std::ceil(_draw_extent.height / 16.0), 1);
    */
}

void Engine::render_geometry(VkCommandBuffer cmd)
{
    //begin a render pass  connected to our draw image
    VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(_draw_image.image_view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    
    VkRenderingInfo render_info = vkinit::rendering_info(_draw_extent, &color_attachment, nullptr);
    vkCmdBeginRendering(cmd, &render_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _mesh_pipeline);
    
    //set dynamic viewport and scissor
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = _draw_extent.width;
    viewport.height = _draw_extent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _draw_extent.width;
    scissor.extent.height = _draw_extent.height;

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _mesh_pipeline);

    GPUDrawPushConstants push_constants;
    push_constants.world_matrix = glm::mat4{ 1.f };
    push_constants.vertex_buffer = rectangle.vertex_buffer_address;

    vkCmdPushConstants(cmd, _mesh_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);
    vkCmdBindIndexBuffer(cmd, rectangle.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
}

void Engine::run()
{
    fmt::print(fg(fmt::color::orange) | fmt::emphasis::bold, "[ Running ]\n");

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