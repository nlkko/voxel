find_package(Vulkan REQUIRED)

add_library(vkbootstrap STATIC)
add_library(glm INTERFACE)
add_library(vma INTERFACE)
add_library(stb INTERFACE)

add_subdirectory(fmt-11.0.2 EXCLUDE_FROM_ALL fmt)
add_subdirectory(SDL2-2.30.5 EXCLUDE_FROM_ALL SDL)
add_subdirectory(vma-3.1.0 EXCLUDE_FROM_ALL vma)
add_subdirectory(fastgltf-0.7.2 EXCLUDE_FROM_ALL fastgltf)

# VkBootstrap
target_sources(vkbootstrap PRIVATE 
  vkbootstrap/VkBootstrap.h
  vkbootstrap/VkBootstrap.cpp
  )

target_include_directories(vkbootstrap PUBLIC vkbootstrap)
target_link_libraries(vkbootstrap PUBLIC Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)
set_property(TARGET vkbootstrap PROPERTY CXX_STANDARD 20)

target_include_directories(glm INTERFACE glm-1.0.1)

# imgui
add_library(imgui STATIC)
target_include_directories(imgui PUBLIC imgui-1.90.9)

target_sources(imgui PRIVATE 
  imgui-1.90.9/imgui.h
  imgui-1.90.9/imgui.cpp

  imgui-1.90.9/imgui_demo.cpp
  imgui-1.90.9/imgui_draw.cpp
  imgui-1.90.9/imgui_widgets.cpp
  imgui-1.90.9/imgui_tables.cpp

  imgui-1.90.9/backends/imgui_impl_vulkan.cpp
  imgui-1.90.9/backends/imgui_impl_sdl2.cpp
)

target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL2::SDL2)

target_include_directories(stb INTERFACE stb)
