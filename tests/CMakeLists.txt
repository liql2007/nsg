set(CMAKE_CXX_STANDARD 14)

# test_faiss_kmeans_recall
add_executable(test_faiss_kmeans_train test_faiss_kmeans_train.cpp)
target_link_libraries(test_faiss_kmeans_train faiss -ltcmalloc)
target_include_directories(test_faiss_kmeans_train PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

# test_faiss_kmeans_recall
add_executable(test_faiss_kmeans_recall test_faiss_kmeans_recall.cpp)
target_link_libraries(test_faiss_kmeans_recall faiss -ltcmalloc)
target_include_directories(test_faiss_kmeans_recall PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

# test_nsg_multi_build_index
add_executable(test_nsg_multi_build_index test_nsg_multi_build_index.cpp)
target_link_libraries(test_nsg_multi_build_index ${PROJECT_NAME} faiss -ltcmalloc)
target_include_directories(test_nsg_multi_build_index PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_compile_options(test_nsg_multi_build_index PRIVATE -Wno-unused-function -Wno-unused-variable)

# test_nsg_multi_merge_index
add_executable(test_nsg_multi_merge_index test_nsg_multi_merge_index.cpp)
target_link_libraries(test_nsg_multi_merge_index ${PROJECT_NAME} faiss -ltcmalloc)
target_include_directories(test_nsg_multi_merge_index PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
target_compile_options(test_nsg_multi_merge_index PRIVATE -Wno-unused-function -Wno-unused-variable)


add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

