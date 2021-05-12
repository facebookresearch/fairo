
cmake_minimum_required (VERSION 2.8.2)
project (lua)

set(LIBSOURCES
	src/lapi.c
	src/lauxlib.c
	src/lbaselib.c
	src/lcode.c
	src/ldblib.c
	src/ldebug.c
	src/ldo.c
	src/ldump.c
	src/lfunc.c
	src/lgc.c
	src/linit.c
	src/liolib.c
	src/llex.c
	src/lmathlib.c
	src/lmem.c
	src/loadlib.c
	src/lobject.c
	src/lopcodes.c
	src/loslib.c
	src/lparser.c
	src/lstate.c
	src/lstring.c
	src/lstrlib.c
	src/ltable.c
	src/ltablib.c
	src/ltm.c
	src/lundump.c
	src/lvm.c
	src/lzio.c
	src/print.c
)

set(LIBHEADERS
	src/lapi.h
	src/lauxlib.h
	src/lcode.h
	src/ldebug.h
	src/ldo.h
	src/lfunc.h
	src/lgc.h
	src/llex.h
	src/llimits.h
	src/lmem.h
	src/lobject.h
	src/lopcodes.h
	src/lparser.h
	src/lstate.h
	src/lstring.h
	src/ltable.h
	src/ltm.h
	src/lua.h
	src/luaconf.h
	src/lualib.h
	src/lundump.h
	src/lvm.h
	src/lzio.h
)



# Lua needs to be linked dynamically on Windows and statically on *nix, so that LuaRocks work
if (WIN32)
	add_library(lua SHARED ${LIBSOURCES} ${LIBHEADERS})
	set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/Server)

	# Output the DLL into the $/Server folder, so that Cuberite can find it:
	set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/Server)
	SET_TARGET_PROPERTIES(lua PROPERTIES
		ARCHIVE_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_SOURCE_DIR}/Server
		ARCHIVE_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_SOURCE_DIR}/Server
		ARCHIVE_OUTPUT_DIRECTORY_DEBUGPROFILE   ${CMAKE_SOURCE_DIR}/Server
		ARCHIVE_OUTPUT_DIRECTORY_RELEASEPROFILE ${CMAKE_SOURCE_DIR}/Server
		LIBRARY_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_SOURCE_DIR}/Server
		LIBRARY_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_SOURCE_DIR}/Server
		LIBRARY_OUTPUT_DIRECTORY_DEBUGPROFILE   ${CMAKE_SOURCE_DIR}/Server
		LIBRARY_OUTPUT_DIRECTORY_RELEASEPROFILE ${CMAKE_SOURCE_DIR}/Server
		RUNTIME_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_SOURCE_DIR}/Server
		RUNTIME_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_SOURCE_DIR}/Server
		RUNTIME_OUTPUT_DIRECTORY_DEBUGPROFILE   ${CMAKE_SOURCE_DIR}/Server
		RUNTIME_OUTPUT_DIRECTORY_RELEASEPROFILE ${CMAKE_SOURCE_DIR}/Server
	)

	if (MSVC)
		# Remove SCL warnings, we expect this library to have been tested safe
		SET_TARGET_PROPERTIES(
			lua PROPERTIES COMPILE_FLAGS "-D_CRT_SECURE_NO_WARNINGS"
		)
	endif()

	set_target_properties(lua PROPERTIES OUTPUT_NAME "lua51" PREFIX "")

	# NOTE: The DLL for each configuration is stored at the same place, thus overwriting each other.
	# This is known, however such behavior is needed for LuaRocks - they always load "lua5.1.dll" or "lua51.dll"
	# We make it work by compiling to "lua51.dll" and providing a proxy-DLL "lua5.1.dll"
	# See http://lua-users.org/wiki/LuaProxyDllFour for details
else()
	add_library(lua ${LIBSOURCES} ${LIBHEADERS})
endif()


# Tell the compiler to use C++ when compiling the library,
# so that Lua uses C++ exceptions instead of setjmp / longjmp (GH #3678):
SET_SOURCE_FILES_PROPERTIES(${SOURCE} PROPERTIES LANGUAGE CXX)

# Tell Lua what dynamic loader to use (for LuaRocks):
if (UNIX)
	add_definitions(-DLUA_USE_DLOPEN)
	target_link_libraries(lua m ${CMAKE_DL_LIBS})
endif()



# If system-wide Lua is not available, build a local one:
if (NOT(USE_SYSTEM_LUA))
	add_executable(luaexe src/lua.c)
	target_link_libraries(luaexe lua)
	SET_TARGET_PROPERTIES(luaexe PROPERTIES
		ARCHIVE_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_SOURCE_DIR}/Server
		ARCHIVE_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_SOURCE_DIR}/Server
		ARCHIVE_OUTPUT_DIRECTORY_DEBUGPROFILE   ${CMAKE_SOURCE_DIR}/Server
		ARCHIVE_OUTPUT_DIRECTORY_RELEASEPROFILE ${CMAKE_SOURCE_DIR}/Server
		LIBRARY_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_SOURCE_DIR}/Server
		LIBRARY_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_SOURCE_DIR}/Server
		LIBRARY_OUTPUT_DIRECTORY_DEBUGPROFILE   ${CMAKE_SOURCE_DIR}/Server
		LIBRARY_OUTPUT_DIRECTORY_RELEASEPROFILE ${CMAKE_SOURCE_DIR}/Server
		RUNTIME_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_SOURCE_DIR}/Server
		RUNTIME_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_SOURCE_DIR}/Server
		RUNTIME_OUTPUT_DIRECTORY_DEBUGPROFILE   ${CMAKE_SOURCE_DIR}/Server
		RUNTIME_OUTPUT_DIRECTORY_RELEASEPROFILE ${CMAKE_SOURCE_DIR}/Server
	)
	if (MSVC)
		set_target_properties(
			luaexe
			PROPERTIES FOLDER Support
		)
	endif()
endif()
