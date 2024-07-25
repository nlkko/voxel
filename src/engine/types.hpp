#pragma once

#include <vk_mem_alloc.h>

#include <vulkan/vk_enum_string_helper.h>

#include <fmt/core.h>

#define VK_CHECK(x) \
    do { \
        VkResult err = (x); \
        if (err) { \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort(); \
        } \
    } while (0)

struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

// holds the resources needed for a mesh
struct GPUMeshBuffers {
    AllocatedBuffer index_buffer;
    AllocatedBuffer vertex_buffer;
    VkDeviceAddress vertex_buffer_address;
};

// push constants for our mesh object draws
struct GPUDrawPushConstants {
    glm::mat4 world_matrix;
    VkDeviceAddress vertex_buffer;
};