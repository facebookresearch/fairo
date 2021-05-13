cmake_minimum_required (VERSION 2.8)
set(CMAKE_CXX_COMPILER "g++")
project (minecraft_client)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

find_library(glog REQUIRED)
find_library(gflags REQUIRED)
find_library(z REQUIRED)
find_library(Boost REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Werror -O3")

set(LINKS gflags glog z)
file(GLOB SOURCES "src/*.cpp")

add_subdirectory(pybind11)
pybind11_add_module(mc_agent ${SOURCES})
target_link_libraries(mc_agent ${LINKS})
set_target_properties(mc_agent PROPERTIES OUTPUT_NAME mc_agent)

add_custom_target(run
	COMMAND python agent/mc_agent.py
	DEPENDS mc_agent
)
