#include <compress/stream_vbyte/StreamVByte.h>
#include <compress/stream_vbyte/StreamVByteDefine.h>
#include <cstring>
#include <cassert>

#ifdef __SSE3__
#include <immintrin.h>
#endif

#if __BYTE_ORDER != __LITTLE_ENDIAN
static_assert(false, "only support LITTLE_ENDIA cpu");
#endif

namespace compress {

namespace {

inline uint8_t controlCode(uint32_t val) {
  assert(val != 0);
  int r;
  __asm__("bsrl %1,%0" : "=&r" (r) : "rm" (val));
  return r >> 3;
}

inline uint8_t encodeInteger(uint32_t val, uint8_t **dp) {
  uint8_t code = controlCode(val);
  *reinterpret_cast<uint32_t*>(*dp) = val;
  *dp += code + 1;
  return code;
}

std::size_t encodeScalar(const uint32_t* in, uint32_t count, uint8_t* out) {
  std::size_t controlBytes = (count + 3) >> 2;
  uint8_t *cp = out;
  uint8_t *dp = out + controlBytes;

#if 0
  uint8_t shift = 0;
  uint8_t key = 0;
  for (uint32_t c = 0; c < count; c++) {
    if (shift == 8) {
      shift = 0;
      *cp++ = key;
      key = 0;
    }
    uint8_t code = encodeInteger(in[c], &dp);
    key |= code << shift;
    shift += 2;
  }
  if (count > 0) {
    *cp = key;
  }
#else
  for (auto end = in + (count & ~3); in != end; in += 4) {
    uint8_t control = 0;
    control |= encodeInteger(in[0], &dp);
    control |= encodeInteger(in[1], &dp) << 2;
    control |= encodeInteger(in[2], &dp) << 4;
    control |= encodeInteger(in[3], &dp) << 6;
    *cp++ = control;
  }
  count &= 3;
  if (count > 0) {
    uint8_t control = encodeInteger(in[0], &dp);
    if (count > 1) {
      control |= encodeInteger(in[1], &dp) << 2;
    }
    if (count > 2) {
      control |= encodeInteger(in[2], &dp) << 4;
    }
    *cp = control;
  }
#endif
//  printf("%u, %u, %u, %u\n", in[0], in[1], in[2], in[3]);
//  printf("%04x, %04x, %04x, %04x\n", in[0], in[1], in[2], in[3]);
//  printf("%02x\n", *out);
//  for (int i = 0; i < 16; ++i) {
//    printf("%02x ", *(out + controlBytes + i));
//  }
//  printf("\n");
  return dp - out;
}

inline uint32_t decodeInteger(const uint8_t** dp, uint8_t code) {
#if 0
  uint32_t val = 0;
  auto bytes = code + 1;
  memcpy(&val, *dp, bytes); // assumes little endian
  *dp += bytes;
  return val;
#else
  const uint8_t *dataPtr = *dp;
  uint32_t val;

  if (code == 0) { // 1 byte
    val = (uint32_t)*dataPtr;
    dataPtr += 1;
  } else if (code == 1) { // 2 bytes
    val = *(uint16_t*)dataPtr;
    dataPtr += 2;
  } else if (code == 2) { // 3 bytes
    val = 0;
    memcpy(&val, dataPtr, 3); // assumes little endian
    dataPtr += 3;
  } else { // code == 3
    val = *(uint32_t*)dataPtr;
    dataPtr += 4;
  }
  *dp = dataPtr;
  return val;
#endif
}

std::size_t decodeScalar(const uint8_t* in, uint32_t count, uint32_t* out) {
  std::size_t controlBytes = (count + 3) >> 2;
  const uint8_t* cp = in;
  const uint8_t* dp = in + controlBytes;

#if 0
  uint8_t key = 0;
  for (uint32_t c = 0; c < count; ++c, key >>= 2) {
    if ((c & 3) == 0) {
      key = *cp++;
    }
    *out++ = decodeInteger(&dp, key & 0x3);
  }
#else
  auto end = count & (~3);
  for (uint32_t c = 0; c < end; c += 4) {
    uint8_t control = *cp++;
    *out++ = decodeInteger(&dp, control & 0x3);
    *out++ = decodeInteger(&dp, (control >> 2) & 0x3);
    *out++ = decodeInteger(&dp, (control >> 4) & 0x3);
    *out++ = decodeInteger(&dp, control >> 6);
  }
  end = count & 3;
  uint8_t control = *cp;
  for (uint32_t i = 0; i < end; ++i, control >>= 2) {
    *out++ = decodeInteger(&dp, control & 0x3);
  }
#endif
//  if (count > 0) {
//    out -= count;
//    printf("decode:\n");
//    printf("%u, %02x\n", key, key);
//    printf("%u\n", out[0]);
//  }
  return dp - in;
}


#ifdef __SSE3__
std::size_t encodeSSE3(const uint32_t* in, uint32_t count, uint8_t* out) {
  std::size_t controlBytes = (count + 3) >> 2;
  uint8_t *cp = out;
  uint8_t *dp = out + controlBytes;

  __m128i mask01 = _mm_set1_epi8(0x01);
  __m128i mask7F00 = _mm_set1_epi16(0x7F00);

  for (auto end = in + (count & ~7); in != end; in += 8) {
    __m128i r0, r1, r2, r3;

    r0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in));
    r1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in + 4));
    // integer(4 bytes) -> 2 bytes (significant bit is 0/1 of every byte)
    r2 = _mm_min_epu8(mask01, r0); // 0xXX -> 0x01, 0x00 -> 0x00
    r3 = _mm_min_epu8(mask01, r1);
    // 0x01XX -> 0xFF; 0x0001 -> 0x01, 0x0000 -> 0x00
    r2 = _mm_packus_epi16(r2, r3);
    // [signed number] 0x01FF -> 0x0101, (0xFFXX, 0x0101, 0x0100, 0x00XX)
    r2 = _mm_min_epi16(r2, mask01);
    // 0xFFXX-> 0xFFFF, 0x0101 -> 0x8001, 0x0100 -> 0x8000, 0x00XX -> 0x7FXX
    r2 = _mm_adds_epu16(r2, mask7F00);
    uint16_t twoControl = _mm_movemask_epi8(r2);

    r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                           &internal::shufTable[(twoControl << 4) & 0x03F0]));
    r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                           &internal::shufTable[(twoControl >> 4) & 0x03F0]));
    r0 = _mm_shuffle_epi8(r0, r2);
    r1 = _mm_shuffle_epi8(r1, r3);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dp), r0);
    dp += internal::lengthTable[twoControl & 0xFF];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dp), r1);
    dp += internal::lengthTable[twoControl >> 8];

    *reinterpret_cast<uint16_t*>(cp) = twoControl;
    cp += 2;
  }

  uint32_t control = 0;
  auto leftCount = count & 7;
  for (uint32_t i = 0; i < leftCount; i++) {
    uint8_t code = encodeInteger(*in++, &dp);
    control |= code << (i + i);
  }
  std::memcpy(cp, &control, (leftCount + 3) >> 2); // assumes little endian

//  if (count > 0) {
//    in -= count;
//    printf ("encode:\n");
//    printf("%u, %u, %u, %u\n", in[0], in[1], in[2], in[3]);
//    printf("%04x, %04x, %04x, %04x\n", in[0], in[1], in[2], in[3]);
//    printf("%02x\n", *out);
//    for (int i = 0; i < 16; ++i) {
//      printf("%02x ", *(out + controlBytes + i));
//    }
//    printf("\n");
//  }

  return dp - out;
}

std::size_t decodeSSE(const uint8_t* in, uint32_t count, uint32_t* out) {
  std::size_t controlBytes = (count + 3) >> 2;
  const uint8_t* cp = in;
  const uint8_t* dp = in + controlBytes;

  auto fullControlBytes = count >> 2;
  for (uint32_t i = 0; i < fullControlBytes; ++i, out += 4) {
    auto key = *cp++;
    __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dp));
    __m128i shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                                        internal::shuffleTable[key]));
    data = _mm_shuffle_epi8(data, shuffle);
    _mm_storeu_si128((__m128i*)out, data);
    dp += internal::lengthTable[key];
  }

  count &= 3;
  auto control = *cp;
  for (uint32_t i = 0; i < count; ++i, control >>= 2) {
    *out++ = decodeInteger(&dp, control & 3);
  }
  return dp - in;
}

#endif // __SSE3__

}


std::size_t StreamVByte::encode(const uint32_t* in, uint32_t count,
                                uint8_t* out) {
  return encodeScalar(in, count, out);
}

std::size_t StreamVByte::encodeSIMD(const uint32_t* in, uint32_t count,
                                    uint8_t* out) {
#ifdef __SSE3__
  return encodeSSE3(in, count, out);
#else
  return 0;
#endif
}

std::size_t StreamVByte::decode(const uint8_t* in, uint32_t count,
                                uint32_t* out) {
  return decodeScalar(in, count, out);
}

std::size_t StreamVByte::decodeSIMD(const uint8_t* in, uint32_t count,
                                    uint32_t* out) {
  return decodeSSE(in, count, out);
}

}
