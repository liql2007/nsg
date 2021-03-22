#ifndef EFANNA2E_STREAMVBYTE_H
#define EFANNA2E_STREAMVBYTE_H

#include <cstdint>
#include <cstdlib>

namespace compress {

class StreamVByte {
public:
  // returns the number of bytes written.
  static std::size_t encode(const uint32_t* in, uint32_t count, uint8_t* out);

  // returns the number of bytes read.
  static std::size_t decode(const uint8_t* in, uint32_t count, uint32_t* out);

  static inline std::size_t maxCompressedBytes(uint32_t count) {
    size_t controlBytes = (count + 3) >> 2;
    size_t maxDataBytes = sizeof(uint32_t) * count;
    return controlBytes + maxDataBytes;
  }
};

class StreamVByteSIMD {
public:
  // returns the number of bytes written.
  static std::size_t encode(const uint32_t* in, uint32_t count, uint8_t* out);

  // returns the number of bytes read.
  static std::size_t decode(const uint8_t* in, uint32_t count, uint32_t* out);

  static inline std::size_t maxCompressedBytes(uint32_t count) {
    size_t controlBytes = (count + 3) >> 2;
    size_t maxDataBytes = sizeof(uint32_t) * count;
    return controlBytes + maxDataBytes;
  }
};

}

#endif //EFANNA2E_STREAMVBYTE_H
