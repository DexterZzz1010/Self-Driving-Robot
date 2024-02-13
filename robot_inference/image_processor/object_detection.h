#ifndef OBJECT_DETECTION_H_
#define OBJECT_DETECTION_H_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "detection_engine.h"
#include "tracker.h"
#include "camera_model.h"
#include "depth_engine.h"

class ObjectDetection {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

public:
    ObjectDetection(): time_pre_process_(0), time_inference_(0), time_post_process_(0) {}
    ~ObjectDetection() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& mat, const cv::Mat& mat_transform, CameraModel& camera);
    bool Draw(cv::Mat& mat, const DepthEngine::Result& depth_result,cv::Mat& mat_topview);
    double GetTimePreProcess() { return time_pre_process_; };
    double GetTimeInference() { return time_inference_; };
    double GetTimePostProcess() { return time_post_process_; };

private:
    cv::Scalar GetColorForId(int32_t id);

private:
    DetectionEngine detection_engine_;
    Tracker tracker_;
    cv::Rect roi_;

    double time_pre_process_;    // [msec]
    double time_inference_;      // [msec]
    double time_post_process_;   // [msec]
};

#endif
