# FAQ for Mac Users

## How to compile in OSX

This section is mainly used as a workaround if you are using mac and having difficulty compiling this project. If you can not build Cuberite and the C++ Minecraft client using ```make``` under root directory, we suggest you compile them separately using the following steps:

1. Build Cuberite server
2. Build C++ Minecraft client
3. Build the rest of the system (log_render, render_view and schematic_convert)

### Build Cuberite server

The easiest way to build Cuberite server is to use its EasyInstall script. This script will download the correct binary from the project site:
 
```
cd cuberite
curl -sSfL https://download.cuberite.org | sh
```
 
### Build C++ Minecraft client
 
To build Minecraft client, run
 
```
cd client
cmake . && make
```
 
### Build the rest of the system
 
Remove the first two targets from ```all``` in [Makefile](Makefile) since they are used to build Cuberite server and C++ Minecraft client which are already built in step 1 & 2. To be specific, replace line#10 of [Makefile](Makefile) from:
 
```
all: cuberite client log_render render_view schematic_convert
```
 
to:
 
```
all: log_render render_view schematic_convert
```
 
Then run

```
make
```

## Common Errors
 
There might be several errors depending on your compiler version. We list some common errors and the ways to resolve them.
 
- **'glog header file can not be found'** when build C++ Minecraft client

    Replace the content of [client/CMakeLists.txt](client/CMakeLists.txt) with the following content:
    
    ```
    cmake_minimum_required (VERSION 2.8)
    set(CMAKE_CXX_COMPILER "g++")
    project (minecraft_client)
    
    find_package(glog REQUIRED)
    find_library(gflags REQUIRED)
    find_library(z REQUIRED)
    find_library(Boost REQUIRED)
    
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Werror -O3")
    
    set(LINKS gflags glog z)
    file(GLOB SOURCES "src/*.cpp")
    
    add_subdirectory(pybind11)
    pybind11_add_module(agent ${SOURCES})
    target_link_libraries(agent ${LINKS})
    target_link_libraries(agent blog)
    set_target_properties(agent PROPERTIES OUTPUT_NAME ../python/agent)
    
    add_custom_target(run
    	COMMAND python agent/agent.py
    	DEPENDS agent
    )
    ```
    
    Then run
    
    ```
    cd client
    cmake . && make
    ```
    
    
- **'optional' file not found'** when build C++ Minecraft client
    
    Switch Xcode to Version 10 or higher, then run

    ```
    cd client
    cmake . && make
    ```
    
- **'_pickle.UnpicklingError: invalid load key, 'v'.'** when you try to start the V0 agent

    Run
    
    ```
    git lfs pull
    ```
    
    then
    
    ```
    python ./python/craftassist/craftassist_agent.py
    ```

- **'glog/logging.h:721:32: error: comparison of integers of different signs: 'const long' and 'const unsigned long''** when build rest of the system
    
    Remove '-Werror' from COMPILE_FLAGS in [Makefile](Makefile). To be specific, replace line#7 of [Makefile](Makefile) from:

    ```
    COMPILE_FLAGS=-std=c++17 -Wall -Wextra -Werror -O3
    ```
    
    to:
    
    ```
    COMPILE_FLAGS=-std=c++17 -Wall -Wextra -O3
    ```












