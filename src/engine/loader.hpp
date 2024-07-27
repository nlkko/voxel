#pragma once

#include "types.hpp"

#include <unordered_map>
#include <filesystem>

//forward declaration
class Engine;

struct GeoSurface {
    uint32_t start_index;
    uint32_t count;
};

struct MeshAsset {
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers mesh_buffers;
};

std::optional<std::vector<std::shared_ptr<MeshAsset>>> load_glTF_meshes(Engine* engine, std::filesystem::path path);
