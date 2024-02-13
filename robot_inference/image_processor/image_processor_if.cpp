/*** Include ***/
/* for general */
#include <cstdint>
#include <memory>

/* for My modules */
#include "image_processor.h"
#include "image_processor_if.h"


/*** Macro ***/
#define TAG "ImageProcessorIf"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Global variable ***/


/*** Function ***/
std::unique_ptr<ImageProcessorIf> ImageProcessorIf::Create()
{
    std::unique_ptr<ImageProcessorIf> ret(new ImageProcessor());
    return ret;
}
