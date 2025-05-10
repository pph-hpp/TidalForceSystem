cmake_minimum_required(VERSION 3.18)  # 确保使用较新的 CMake 版本
project(TidalForceSimulator)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA支持
enable_language(CUDA)


# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # 根据实际GPU调整

# 收集源文件

file(GLOB_RECURSE CPP_SOURCES
    "Render/*.h"
    "Render/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.[ch]pp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.[ch]h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)

# 创建可执行文件
add_executable(${CMAKE_PROJECT_NAME} ${CPP_SOURCES})

# 确保 CUDA 代码正确使用 nvcc 编译
# set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)


# 添加头文件目录
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/glfw3.lib
    ${CMAKE_SOURCE_DIR}/lib/glew32.lib
    ${CMAKE_SOURCE_DIR}/lib/glm.lib
)

# 设置CUDA特定的编译选项
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 调试配置
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(${PROJECT_NAME} PROPERTIES 
        CUDA_FLAGS "${CUDA_FLAGS} -G"
    )
endif()


# 确保可执行输出目录存在
add_custom_command(
  TARGET ${PROJECT_NAME}
  PRE_BUILD                       # VS 专用：在任何编译之前执行
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
)

# 在 PRE_BUILD 事件中拷贝所有运行时依赖
add_custom_command(
  TARGET ${PROJECT_NAME}
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/lib/glew32.dll"
          "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
  COMMENT "Copying GLFW and GLEW DLLs before build"
)