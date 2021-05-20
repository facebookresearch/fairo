SHELL=/bin/bash

.PHONY: cuberite clang-format client log_render render_view

CC?=gcc
CXX?=g++
COMMON_CLIENT_CPP=client/src/block_map.cpp client/src/game_state.cpp client/src/graphics.cpp client/src/packet_writer.cpp client/src/types.cpp client/src/util.cpp client/src/nbt_tag.cpp
COMMON_LOGGING_CPP=logging/src/anvil_reader.cpp logging/src/logging_reader.cpp
COMPILE_FLAGS=-std=c++17 -Wall -Wextra -Werror -O3
LINK_FLAGS=-lglog -lgflags -lz -pthread
ifeq ($(CMAKE_PREFIX_PATH),)
INCLUDE_FLAGS:=
LIBDIR_FLAGS:=
else
INCLUDE_FLAGS:=-I $(CMAKE_PREFIX_PATH)/include
LIBDIR_FLAGS:=-L $(CMAKE_PREFIX_PATH)/lib
endif

BIN=bin/log_render bin/render_view
CUBERITE_MAKEFILE=cuberite/Makefile
all: cuberite client log_render render_view

cuberite:
	cd server/cuberite && \
	mkdir -p Release && \
	cd Release && \
	cmake -DCMAKE_BUILD_TYPE=RELEASE .. && \
	make -j`nproc`

client:
	cd client && \
	rm -rf build && \
	mkdir -p build && \
	cd build && \
	cmake .. && \
	make

log_render:
	${CXX} ${COMPILE_FLAGS} \
		${INCLUDE_FLAGS} ${LIBDIR_FLAGS} \
		${COMMON_CLIENT_CPP} ${COMMON_LOGGING_CPP} logging/src/log_render.cpp \
		-o bin/log_render \
		${LINK_FLAGS} -Wno-sign-compare -Wno-deprecated-copy

render_view:
	${CXX} ${COMPILE_FLAGS} \
		${INCLUDE_FLAGS} ${LIBDIR_FLAGS} \
		${COMMON_CLIENT_CPP} ${COMMON_LOGGING_CPP} logging/src/render_view.cpp \
		-o bin/render_view \
		${LINK_FLAGS} -Wno-sign-compare -Wno-deprecated-copy

clang-format:
	bin/clang-format

.PHONY: clean

clean:
	find . -name '*.so' -delete
	find . -name 'cmake_install.cmake' -delete
	find . -name 'CMakeCache.txt' -delete
	if [ -f "$(CUBERITE_MAKEFILE)" ]; then \
		cd server/cuberite && make clean && cd ../..; \
	fi
	rm -rf bin/log_render bin/render_view
	rm -rf client/pybind11/CMakeFiles
	rm -rf client/pybind11/Makefile
	rm -rf client/CMakeFiles
	rm -rf client/Makefile
	rm -rf cuberite/Release
