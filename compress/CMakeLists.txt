set(CMAKE_CXX_STANDARD 14)

file(GLOB_RECURSE CPP_SOURCES *.cpp)

add_library(compress ${CPP_SOURCES})
target_include_directories(compress PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
#target_compile_options(compress PRIVATE -Wno-sign-compare -Wno-unused-variable)

add_library(compress_s STATIC ${CPP_SOURCES})
target_include_directories(compress_s PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
#target_compile_options(compress_s PRIVATE -Wno-sign-compare -Wno-unused-variable)