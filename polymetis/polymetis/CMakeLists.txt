# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.14)
project(polymetis)

set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
set_target_properties(protobuf::libprotobuf PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{CONDA_PREFIX}/include/google/protobuf/")
set_target_properties(protobuf::libprotobuf PROPERTIES
  IMPORTED_LOCATION "$ENV{CONDA_PREFIX}/lib/libprotobuf.so")
set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
  INTERFACE_COMPILE_FEATURES cxx_std_11
)

add_subdirectory("torch_isolation/")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

if(DEFINED ENV{PREFIX})
  message("Using $ENV{PREFIX} as output dir")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY $ENV{PREFIX}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY $ENV{PREFIX}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $ENV{PREFIX}/bin)
endif()

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

### eigen3
find_package(Eigen3 REQUIRED)


# Find protoc
set(_PROTOBUF_PROTOC $ENV{CONDA_PREFIX}/bin/protoc)

# Find gRPC
set(gRPC_DIR $ENV{CONDA_PREFIX}/lib/cmake/grpc/)
find_package(gRPC REQUIRED)
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)

# Find gRPC plugins
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE $ENV{CONDA_PREFIX}/bin/grpc_cpp_plugin)
find_program(_GRPC_PYTHON_PLUGIN_EXECUTABLE $ENV{CONDA_PREFIX}/bin/grpc_python_plugin)

# Find spdlog (C++ logger)
if(NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()

# Define proto directories
get_filename_component(loader_proto "protos/polymetis.proto" ABSOLUTE)
get_filename_component(loader_proto_path "${loader_proto}" PATH)

# Define generated sources
set(loader_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/polymetis.pb.cc")
set(loader_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/polymetis.pb.h")
set(loader_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/polymetis.grpc.pb.cc")
set(loader_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/polymetis.grpc.pb.h")

# Use protoc to generate C++
add_custom_command(
    OUTPUT
      ${loader_proto_srcs}
      ${loader_proto_hdrs}
      ${loader_grpc_srcs}
      ${loader_grpc_hdrs}
    COMMAND
      ${_PROTOBUF_PROTOC}
    ARGS
      --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
      --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
      --grpc_python_out "${CMAKE_CURRENT_BINARY_DIR}"
      --python_out "${CMAKE_CURRENT_BINARY_DIR}"
      -I "${loader_proto_path}"
      -I "$ENV{CONDA_PREFIX}/include/"
      --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
      --plugin=protoc-gen-grpc_python="${_GRPC_PYTHON_PLUGIN_EXECUTABLE}"
      "${loader_proto}"
    DEPENDS
      "${loader_proto}"
)

add_library(
    generated_grpc_protos
      ${loader_proto_srcs}
      ${loader_proto_hdrs}
      ${loader_grpc_srcs}
      ${loader_grpc_hdrs}
)
target_link_libraries(generated_grpc_protos
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
)

include_directories(
    include
    "${CMAKE_CURRENT_BINARY_DIR}"
    "$ENV{CONDA_PREFIX}/include/"
    ${EIGEN3_INCLUDE_DIR}
    ${pinocchio_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_isolation/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_isolation/pinocchio_isolation/include"
)

# Add server
add_library(polymetis_server "src/polymetis_server.cpp")

target_link_libraries(polymetis_server
    generated_grpc_protos
    yaml-cpp
    spdlog::spdlog
)
target_link_libraries(polymetis_server -Wl,--no-as-needed torchscript_pinocchio)
target_link_libraries(polymetis_server -Wl,--no-as-needed torchrot)
target_link_libraries(polymetis_server -Wl,--no-as-needed torch_server_ops)

add_executable(run_server "src/run_server.cpp")
target_link_libraries(run_server polymetis_server)

# Add test client
add_executable(empty_statistics_client "src/clients/empty_statistics_client.cpp"
)
target_link_libraries(empty_statistics_client
    generated_grpc_protos
    yaml-cpp
    spdlog::spdlog
)

if(BUILD_ALLEGRO)
  add_library(allegro_hand
    src/clients/allegro_hand_client/allegro_hand.cpp
    src/clients/allegro_hand_client/event_watcher.cpp
    src/clients/allegro_hand_client/pcan_netdev_interface.cpp)

  target_link_libraries(allegro_hand
    yaml-cpp
    spdlog::spdlog
    )

  add_executable(allegro_hand_client
    src/clients/allegro_hand_client/allegro_hand_client.cpp
    )

  target_link_libraries(allegro_hand_client
    allegro_hand
    generated_grpc_protos
    )

  if(BUILD_TESTS)
    find_package(Git REQUIRED QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- tests/cpp/fake_clock
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND_ERROR_IS_FATAL ANY)

    add_library(fake_clock tests/cpp/fake_clock/fake_clock.cc)
    add_executable(test_allegro_hand_client
      tests/cpp/test_allegro_hand_client.cpp
      )

    target_include_directories(test_allegro_hand_client PRIVATE
      src/clients/allegro_hand_client
      ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
      )

    target_link_libraries(test_allegro_hand_client
      allegro_hand
      fake_clock
      gtest
      gtest_main
      )
  endif()
endif()

if(BUILD_FRANKA)
  if(NOT DEFINED Franka_DIR)
    set(Franka_DIR "src/clients/franka_panda_client/third_party/libfranka/build")
  endif()
  message("Compiling Franka client with libfranka from ${Franka_DIR}")
  find_package(Franka REQUIRED)

  include_directories(
    ${Franka_INCLUDE_DIRS}
  )

  # Add Franka clients
  add_executable(franka_panda_client
      src/clients/franka_panda_client/franka_panda_client.cpp
  )
  target_link_libraries(franka_panda_client
      generated_grpc_protos
      Franka::Franka
      yaml-cpp
      spdlog::spdlog
  )

  add_executable(franka_hand_client 
      src/clients/franka_panda_client/franka_hand_client.cpp
  )
  target_link_libraries(franka_hand_client
      generated_grpc_protos
      Franka::Franka
      yaml-cpp
      spdlog::spdlog
  )
endif()


if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    # Specify the commit you depend on and update it regularly.
    URL https://github.com/google/googletest/archive/23ef29555ef4789f555f1ba8c51b4c52975f0907.zip
  )
  FetchContent_MakeAvailable(googletest)

  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  add_executable(test_server ./tests/cpp/test_server.cpp)
  target_link_libraries(test_server polymetis_server gtest)
endif()

if(BUILD_DOCS)
  find_package(Doxygen)

  # set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/../docs/Doxyfile)

  # note the option ALL which allows to build the docs together with the application
  add_custom_target( doc_doxygen ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../docs/
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM )

endif()
