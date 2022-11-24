
#ifndef DALI_IMAGE_L3_H_
#define DALI_IMAGE_L3_H_

#include "dali/core/common.h"
#include "dali/image/generic_image.h"

namespace dali {

bool CheckIsL3Image(const uint8_t *l3image, int size);

/**
 * New lightweight lossless image decoding is performed using OpenCV, thus it's the same as Generic decoding
 */
class L3Image final : public GenericImage {
 public:
  L3Image(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);

  ~L3Image() override = default;

 private:
  Shape PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const override;
};

}  // namespace dali

#endif  // DALI_IMAGE_L3_H_

