cmake_minimum_required (VERSION 3.0)
project (minecraft_client)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)
include(${CMAKE_SOURCE_DIR}/cmake/Modules/Findglog.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Modules/Findgflags.cmake)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_library(LIBZ_LIBRARY z REQUIRED)
find_package(Boost REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -O3")

include_directories(${GLOG_INCLUDE_DIR})
include_directories(${GFLAGS_INCLUDE_DIR})

file(GLOB SOURCES "src/*.cpp")

MESSAGE(STATUS "LIBS: ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${LIBZ_LIBRARY}")

add_subdirectory(pybind11)
pybind11_add_module(mc_agent ${SOURCES})
target_link_libraries(mc_agent PUBLIC ${GFLAGS_LIBRARY} ${GLOG_LIBRARY} ${LIBZ_LIBRARY})
set_target_properties(mc_agent PROPERTIES OUTPUT_NAME ${CMAKE_CURRENT_LIST_DIR}/../mc_agent)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-deprecated-copy")

add_custom_target(run
	COMMAND python agent/mc_agent.py
	DEPENDS mc_agent
)
