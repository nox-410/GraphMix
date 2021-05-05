cmake_minimum_required(VERSION 3.18)

#set(CMAKE_CXX_COMPILER "/usr/bin/g++-9")

project(GraphMix CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

include(FetchContent) # download third_party

add_subdirectory(${PROJECT_SOURCE_DIR}/graphmix)

enable_testing()
ADD_TEST(NAME graph COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/test_graph.py)
ADD_TEST(NAME pullnode COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/test_pull_node.py
    --config ${PROJECT_SOURCE_DIR}/config/test_config.yml)
ADD_TEST(NAME basic_sample COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/test_base_sampling.py
    --config ${PROJECT_SOURCE_DIR}/config/test_config.yml)
ADD_TEST(NAME remote_sample COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/test_remote_sampling.py
    --config ${PROJECT_SOURCE_DIR}/config/test_config.yml)
ADD_TEST(NAME sampler COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/test_samplers.py
    --config ${PROJECT_SOURCE_DIR}/config/test_config.yml)
ADD_TEST(NAME burst COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/test_burst.py
    --config ${PROJECT_SOURCE_DIR}/config/test_config.yml)

add_custom_target(graphmix_test_data COMMAND
    python3 -m graphmix.partition -d Cora -n 4 --nodeid -p ${PROJECT_SOURCE_DIR}/data)
