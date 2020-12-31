set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE CPP_SOURCES *.cpp)

add_library(faiss ${CPP_SOURCES})
target_include_directories(faiss PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_compile_options(faiss PRIVATE -Wno-sign-compare -Wno-unused-variable)

add_library(faiss_s STATIC ${CPP_SOURCES})
target_include_directories(faiss_s PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_compile_options(faiss_s PRIVATE -Wno-sign-compare -Wno-unused-variable)

#install()