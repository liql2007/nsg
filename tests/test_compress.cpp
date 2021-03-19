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

void genData(std::size_t size, std::size_t maxGroupSize,
             std::vector<std::uint32_t>* vec, std::vector<std::size_t>* group) {
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
  group->clear();
  std::size_t index = 0;
  group->reserve(3 * size / maxGroupSize);
  for (int i = 0; i <= 32 && index + i < size; ++i, index += i) {
    group->push_back(i);
  }
  while (index < size) {
    std::size_t count = std::abs(rand()) % maxGroupSize;
    if (count < 20) {
      count += 20;
    }
    count = std::min(count, size - index);
    group->push_back(count);
    index += count;
  }
}

void testSVB() {
  constexpr std::size_t maxGroupSize = 111;
  std::size_t size = 1024LL * 1024 * 1024;
  /** cost time:
   *  encodeSIMD: 1.30, encode:2.11
   *  encodeSIMD + decodeSIMD: 2.09
   *  encodeSIMD + decode: 8.75
   */
  // std::size_t size = 1024LL * 1024;
  std::vector<std::uint32_t> vec;
  std::vector<std::size_t> group;
  genData(size, maxGroupSize, &vec, &group);

  std::size_t index = 0;
  auto encodeBuffSize = StreamVByte::maxCompressedBytes(maxGroupSize);
  std::vector<uint8_t> encodeBuffer1(encodeBuffSize);
  std::vector<uint32_t> decodeBuffer1(maxGroupSize);

#if 1
  std::vector<uint8_t> encodeBuffer2(encodeBuffSize);
  std::vector<uint32_t> decodeBuffer2(maxGroupSize);

  for (auto count : group) { // warn up & test accuracy
    auto es1 = StreamVByte::encode(&vec[index], count, encodeBuffer1.data());
    auto es2 = StreamVByte::encodeSIMD(&vec[index], count, encodeBuffer2.data());
    auto ds1 = StreamVByte::decode(encodeBuffer1.data(), count,
                                   decodeBuffer1.data());
    auto ds2 = StreamVByte::decodeSIMD(encodeBuffer2.data(), count,
                                       decodeBuffer2.data());
    if (ds1 != es1 || es2 != es1 || ds2 != ds1) {
      printf("[%zu] size mismatch, encode [%zu:%zu], decode [%zu:%zu]",
             index, es1, es2, ds1, ds2);
      exit(-1);
    }
    for (decltype(count) i = 0; i < count; ++i) {
      if (vec[index + i] != decodeBuffer1[i]) {
        printf("decode value error: [%zu, %zu], [%u, %u]\n",
               index + i, i, vec[index + i], decodeBuffer1[i]);
        exit(-1);
      }
    }
    for (decltype(es1) i = 0; i < es1; ++i) {
      if (encodeBuffer1[i] != encodeBuffer2[i]) {
        printf("new encoded [%zu:%zu] value error, [%u, %u]",
               index, i, encodeBuffer1[i], encodeBuffer2[i]);
        exit(-1);
      }
    }
    for (std::size_t i = 0; i < count; ++i) {
      if (decodeBuffer1[i] != decodeBuffer2[i]) {
        printf("new decoded [%zu:%zu] value error, [%u, %u]",
               index, i, decodeBuffer1[i], decodeBuffer2[i]);
        exit(-1);
      }
    }
    index += count;
  }
#endif

  uint32_t runTimes = 4;
  for (uint32_t i = 0; i < runTimes; ++i) {
    auto bb = std::chrono::high_resolution_clock::now();
    index = 0;
    for (auto count : group) {
      StreamVByte::encodeSIMD(&vec[index], count, encodeBuffer1.data());
      // StreamVByte::encode(&vec[index], count, encodeBuffer1.data());
      StreamVByte::decodeSIMD(encodeBuffer1.data(), count, decodeBuffer1.data());
      // StreamVByte::decode(encodeBuffer1.data(), count, decodeBuffer1.data());
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