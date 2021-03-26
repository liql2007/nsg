set(CMAKE_CXX_STANDARD 14)

# test_faiss_kmeans_recall
add_executable(test_faiss_kmeans_train test_faiss_kmeans_train.cpp)
target_link_libraries(test_faiss_kmeans_train faiss -ltcmalloc)

# test_faiss_kmeans_recall
add_executable(test_faiss_kmeans_recall test_faiss_kmeans_recall.cpp)
target_link_libraries(test_faiss_kmeans_recall faiss -ltcmalloc)

# test_nsg_multi_build_index
add_executable(test_nsg_multi_build_index test_nsg_multi_build_index.cpp)
target_link_libraries(test_nsg_multi_build_index ${PROJECT_NAME} faiss compress -ltcmalloc)
target_compile_options(test_nsg_multi_build_index PRIVATE -Wno-unused-function -Wno-unused-variable)

# test_create_overlapping_shards
add_executable(test_create_overlapping_shards test_create_overlapping_shards.cpp)
target_link_libraries(test_create_overlapping_shards ${PROJECT_NAME} faiss compress -ltcmalloc)

# test_nsg_multi_merge_index
add_executable(test_nsg_multi_merge_index test_nsg_multi_merge_index.cpp)
target_link_libraries(test_nsg_multi_merge_index ${PROJECT_NAME} faiss compress -ltcmalloc)
target_compile_options(test_nsg_multi_merge_index PRIVATE -Wno-unused-function -Wno-unused-variable)

# test_create_ground_truth
add_executable(test_create_ground_truth test_create_ground_truth.cpp)
target_link_libraries(test_create_ground_truth ${PROJECT_NAME} compress -ltcmalloc)

# test_create_shards
add_executable(test_create_shards test_create_shards.cpp)
target_link_libraries(test_create_shards ${PROJECT_NAME} compress -ltcmalloc)

# test_sample_data
add_executable(test_sample_data test_sample_data.cpp)
target_link_libraries(test_sample_data ${PROJECT_NAME} compress -ltcmalloc)

# test_reordering
add_executable(test_reordering test_reordering.cpp)
target_link_libraries(test_reordering ${PROJECT_NAME} compress -ltcmalloc)

# test_compress
add_executable(test_compress test_compress.cpp)
target_link_libraries(test_compress ${PROJECT_NAME} compress -ltcmalloc)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} compress -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME} compress)

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} compress -ltcmalloc)

add_executable(test_nsg_optimized_multi_search test_nsg_optimized_multi_search.cpp)
target_link_libraries(test_nsg_optimized_multi_search ${PROJECT_NAME} compress -ltcmalloc)

