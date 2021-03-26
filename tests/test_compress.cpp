//
// Created by liql2007 on 2021/3/17.
//
#include <compress/stream_vbyte/StreamVByte.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <chrono>

using namespace compress;

void genData(std::size_t size, std::size_t maxGroupSize, bool sort,
             std::vector<std::uint32_t>* vec,
             std::vector<std::size_t>* group) {
  vec->resize(size);
  srand(12345);
  for (std::size_t i = 0; i < size; ++i) {
    while(true) {
      auto v = std::abs(rand()) & ((1 << 20) - 1);
      if (v != 0) {
        (*vec)[i] = v;
        break;
      }
    }
  }
  auto sortGroup = [&vec, sort](std::size_t index, std::size_t count) {
    if (sort) {
      std::sort(vec->data() + index, vec->data() + index + count);
      for (decltype(count) i = index + 1; i < index + count; ++i) {
        if ((*vec)[i] <= (*vec)[i - 1]) {
          (*vec)[i] = (*vec)[i - 1] + 1;
        }
      }
    }
  };
  group->clear();
  std::size_t index = 0;
  group->reserve(3 * size / maxGroupSize);
  for (int count = 0; count <= 32 && index + count < size; ++count) {
    group->push_back(count);
    sortGroup(index, count);
    index += count;
  }
  while (index < size) {
    std::size_t count = std::abs(rand()) % maxGroupSize;
    if (count < 20) {
      count += 20;
    }
    count = std::min(count, size - index);
    group->push_back(count);
    sortGroup(index, count);
    index += count;
  }
}

template <typename Compressor>
void encodeAndDecode(const std::vector<std::uint32_t>& vec,
                     const std::vector<std::size_t>& group,
                     std::vector<uint8_t>& encodeBuffer,
                     std::vector<uint32_t>& decodeBuffer) {
  std::size_t totalSize = 0;
  std::size_t index = 0;
  for (auto count : group) {
    auto es1 = Compressor::encode(&vec[index], count, encodeBuffer.data());
    totalSize += es1;
    auto ds1 = Compressor::decode(encodeBuffer.data(), count,
                                  decodeBuffer.data());
    if (ds1 != es1) {
      printf("[%zu] size mismatch, encode [%zu], decode [%zu]",
             index, es1, ds1);
      exit(-1);
    }
    for (decltype(count) i = 0; i < count; ++i) {
      if (vec[index + i] != decodeBuffer[i]) {
        printf("decode value error: [%zu, %zu], [%u, %u]\n",
               index, i, vec[index + i], decodeBuffer[i]);
        exit(-1);
      }
    }
    index += count;
  }
  auto oriSize = vec.size() * sizeof(uint32_t);
  printf("compress rate: %.1f%%\n", 100.0 * totalSize / oriSize);
}

void testSVB() {
  constexpr std::size_t maxGroupSize = 111;
  std::size_t size = 1024LL * 1024 * 1024;
  /** cost time(no delta):
   *  encodeSIMD: 1.30, encode:2.11
   *  encodeSIMD + decodeSIMD: 2.09
   *  encodeSIMD + decode: 8.75
   ** cost time(delta):
   *  encodeSIMD: 1.53, encode:2.26, compress rate: 56.9%
   *  encodeSIMD + decodeSIMD: 2.42
   *  encode + decode: 5.68 (branch predicate better)
   */
  // std::size_t size = 1024LL * 1024;
  std::vector<std::uint32_t> vec;
  std::vector<std::size_t> group;
  genData(size, maxGroupSize, true, &vec, &group);

  auto encodeBuffSize = StreamVByte::maxCompressedBytes(maxGroupSize);
  std::vector<uint8_t> encodeBuffer(encodeBuffSize);
  std::vector<uint32_t> decodeBuffer(maxGroupSize);

#if 1
  printf("scalar encode and decode:\n");
  encodeAndDecode<StreamVByte>(vec, group,encodeBuffer, decodeBuffer);
  printf("vector encode and decode:\n");
  encodeAndDecode<StreamVByteSIMD>(vec, group,encodeBuffer, decodeBuffer);
#endif

  // using SVB = StreamVByte;
  using SVB = StreamVByteSIMD;
  uint32_t runTimes = 4;
  for (uint32_t i = 0; i < runTimes; ++i) {
    auto bb = std::chrono::high_resolution_clock::now();
    std::size_t index = 0;
    for (auto count : group) {
      SVB::encode(&vec[index], count, encodeBuffer.data());
      SVB::decode(encodeBuffer.data(), count, decodeBuffer.data());
      index += count;
    }
    auto ee = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = ee - bb;
    std::cout << "[" << i + 1 << "] cost time: " << diff.count() << "\n";
  }
}

int main(int argc, char** argv) {
  testSVB();
}