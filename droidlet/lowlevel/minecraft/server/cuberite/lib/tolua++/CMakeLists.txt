
cmake_minimum_required (VERSION 2.6)
project (tolua++)

include_directories ("${PROJECT_SOURCE_DIR}/../../src/")
include_directories ("${PROJECT_SOURCE_DIR}/include/")
include_directories ("${PROJECT_SOURCE_DIR}/../")
include_directories ("${PROJECT_SOURCE_DIR}")

SET (LIB_SOURCE
	src/lib/tolua_event.c
	src/lib/tolua_is.c
	src/lib/tolua_map.c
	src/lib/tolua_push.c
	src/lib/tolua_to.c
)

add_library(tolualib ${LIB_SOURCE})
target_link_libraries(tolualib lua)

if (${CMAKE_SYSTEM_NAME} STREQUAL "FreeBSD")
	add_flags_lnk(-L/usr/local/lib)
endif()

