
#include "dali/image/l3.h"
#include "dali/core/byte_io.h"

namespace dali {

namespace {

constexpr int kOffsetWidth = sizeof(uint32_t);
constexpr int kOffsetHeight = kOffsetWidth + sizeof(uint32_t);

uint32_t ReadHeight(const uint8_t *data) {
  return ReadValueLE<uint32_t>(data + kOffsetHeight);
}

uint32_t ReadWidth(const uint8_t *data) {
  return ReadValueLE<uint32_t>(data + kOffsetWidth);
}

}  // namespace


bool CheckIsL3Image(const uint8_t *l3image, int size) {
  return ((size > 4) && (l3image[0] == 'L') && (l3image[1] == 'L') &&
          (l3image[2] == 'L') && (l3image[3] == '.'));
}


L3Image::L3Image(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        GenericImage(encoded_buffer, length, image_type) {
}


Image::Shape L3Image::PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const {
  DALI_ENFORCE(encoded_buffer);
  DALI_ENFORCE(length >= 16);

  const int64_t W = ReadWidth(encoded_buffer);
  const int64_t H = ReadHeight(encoded_buffer);

  // L3 currently only support RGB channel format image
  const int64_t C = 3;
  return {H, W, C};
}


}  // namespace dali

