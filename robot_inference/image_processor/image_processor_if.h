#ifndef IMAGE_PROCESSOR_IF_H_
#define IMAGE_PROCESSOR_IF_H_

/* for general */
#include <cstdint>
#include <memory>

namespace cv {
    class Mat;
};

class ImageProcessorIf {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    typedef struct {
        char     work_dir[256];
        int32_t  num_threads;
    } InputParam;

    typedef struct {
        cv::Mat mat_output;
        cv::Mat mat_output_segmentation;
        cv::Mat mat_output_depth;
        cv::Mat mat_output_topview;
	float err;
        double time_pre_process;   // [msec]
        double time_inference;    // [msec]
        double time_post_process;  // [msec]
    } Result;

public:
    static std::unique_ptr<ImageProcessorIf> Create();

public:
    virtual ~ImageProcessorIf() {}
    virtual int32_t Initialize(const InputParam& input_param) = 0;
    virtual int32_t Process(const cv::Mat& mat_original, Result& result) = 0;
    virtual int32_t Finalize(void) = 0;
    virtual int32_t Command(int32_t cmd) = 0;

    virtual void ResetCamera(int32_t width = 0, int32_t height = 0, float fov_deg = 130.0f) = 0;
    virtual void GetCameraParameter(float& focal_length, std::array<float, 3>& real_rvec, std::array<float, 3>& real_tvec, std::array<float, 3>& top_rvec, std::array<float, 3>& top_tvec) = 0;
    virtual void SetCameraParameter(float focal_length, const std::array<float, 3>& real_rvec, const std::array<float, 3>& real_tvec, const std::array<float, 3>& top_rvec, const std::array<float, 3>& top_tvec) = 0;
};

#endif
