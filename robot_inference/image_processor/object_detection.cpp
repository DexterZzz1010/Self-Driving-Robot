/*

Initialize(...)：初始化对象检测器(detection_engine_)，工作目录为work_dir，线程数为num_threads，如果初始化失败则返回kRetErr，否则返回kRetOk。
Finalize()：结束对象检测器(detection_engine_)，如果结束失败则返回kRetErr，否则返回kRetOk。
Process(...)：处理传入的mat图像，将检测出的物体进行跟踪并计算出物体的地平面坐标，并将跟踪结果保存下来，如果处理失败则返回kRetErr，否则返回kRetOk。
Draw(...)：将处理后的跟踪结果画到传入的两张图(mat和mat_topview)上，并显示一些跟踪信息。
GetColorForId(...)：给定一个id，返回一个对应的颜色，id最大为100，超过100的取模。
该类的实现还依赖其他模块：

common_helper.h：一些常用函数的封装。
common_helper_cv.h：一些OpenCV函数的封装。
object_detection.h：一个C++封装的TensorFlow Lite对象检测器(DetectionEngine)。

*/


/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "object_detection.h"


/*** Macro ***/
#define TAG "ObjectDetection"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

static constexpr float kDetectThreshold = 0.6f;
static constexpr double kDepthThreshold = 150;



/*** Function ***/
int32_t ObjectDetection::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    if (detection_engine_.Initialize(work_dir, num_threads) != DetectionEngine::kRetOk) {
        detection_engine_.Finalize();
        return kRetErr;
    }
    return kRetOk;
}

int32_t ObjectDetection::Finalize()
{
    if (detection_engine_.Finalize() != DetectionEngine::kRetOk) {
        return kRetErr;
    }
    return kRetOk;
}


/*
接受3个参数：一个cv::Mat类型的图像，一个cv::Mat类型的变换矩阵和一个CameraModel类型的相机模型。该方法用于目标检测和追踪，将检测到的目标进行跟踪，并转换成世界坐标系上的点。

首先通过检测引擎（DetectionEngine）对图像进行推理，得到检测结果（DetectionEngine::Result），并提取关键信息并存储。

然后基于之前得到的检测结果，将检测到的目标进行跟踪（tracker），并提取对应的每个跟踪器的信息。

接下来，将每个跟踪器的边界框（bbox）的底部中心点从视图坐标（normal view）转换为鸟瞰视图（top view），并将转换后的结果存储在跟踪器的信息中。

最后，将鸟瞰视图中的点转换为世界坐标系中的距离（通过相机模型），并将计算出的世界坐标系中的点坐标放入跟踪器信息的对象点（object_point）中。

最终，该方法返回一个整数状态码，说明方法的执行是否成功。

*/
int32_t ObjectDetection::Process(const cv::Mat& mat, const cv::Mat& mat_transform, CameraModel& camera)
{
    /* Run inference to detect objects */
    DetectionEngine::Result det_result;
    if (detection_engine_.Process(mat, det_result) != DetectionEngine::kRetOk) {
        return kRetErr;
    }
    roi_ = cv::Rect(det_result.crop.x, det_result.crop.y, det_result.crop.w, det_result.crop.h);
    time_pre_process_ = det_result.time_pre_process;
    time_inference_ = det_result.time_inference;
    time_post_process_ = det_result.time_post_process;

    /* Track */
    tracker_.Update(det_result.bbox_list);
    auto& track_list = tracker_.GetTrackList(); //

    /* Convert points from normal view -> top view. Store the results into track data */
    std::vector<cv::Point2f> normal_points;
    std::vector<cv::Point2f> topview_points;
    for (auto& track : track_list) {
        const auto& bbox = track.GetLatestData().bbox;
        normal_points.push_back({ bbox.x + bbox.w / 2.0f, bbox.y + bbox.h + 0.0f }); //计算中心点
    }
    if (normal_points.size() > 0) {
        cv::perspectiveTransform(normal_points, topview_points, mat_transform);
        for (int32_t i = 0; i < static_cast<int32_t>(track_list.size()); i++) {
            auto& track_data = track_list[i].GetLatestData();
            track_data.topview_point.x = static_cast<int32_t>(topview_points[i].x);
            track_data.topview_point.y = static_cast<int32_t>(topview_points[i].y);
        }
    }

    /* Calcualte points in world coordinate (distance on ground plane) */
    std::vector<cv::Point2f> image_point_list;
    std::vector<cv::Point3f> object_point_list;
    for (auto& track : track_list) {
        const auto& bbox = track.GetLatestData().bbox;
        image_point_list.push_back(cv::Point2f(bbox.x + bbox.w / 2.0f, bbox.y + bbox.h + 0.0f));
    }
    camera.ConvertImage2GroundPlane(image_point_list, object_point_list);

    int32_t index = 0;
    for (auto& track : track_list) {
        auto& object_point_track = track.GetLatestData().object_point;
        object_point_track.x = object_point_list[index].x;
        object_point_track.y = object_point_list[index].y;
        object_point_track.z = object_point_list[index].z;
        index++;
    }

    return kRetOk;
}

bool ObjectDetection::Draw(cv::Mat& mat, const DepthEngine::Result& depth_result, cv::Mat& mat_topview)
{
    cv::Mat depth_clone = depth_result.mat_out_ori.clone();
    int32_t crop_w = mat.cols;
    int32_t crop_h = mat.rows;
    float min_x = crop_w*(1-kDetectThreshold)/2;
    float max_x = crop_w*(1+kDetectThreshold)/2;
    cv::Rect roi;
    bool Object_flag = false;

    auto& track_list = tracker_.GetTrackList();

    cv::resize(depth_clone, depth_clone,cv::Size(crop_w , crop_h));

    /* Draw on NormalView */
    cv::rectangle(mat, roi_, CommonHelper::CreateCvColor(0, 0, 0), 2);
    int32_t det_num = 0;
    for (auto& track : track_list) {
        if (track.GetUndetectedCount() > 0) continue;
        const auto& bbox = track.GetLatestData().bbox_raw;
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CommonHelper::CreateCvColor(0, 0, 0), 1);
        det_num++;
    }
    CommonHelper::DrawText(mat, "DET: " + std::to_string(det_num) + ", TRACK: " + std::to_string(tracker_.GetTrackList().size()), cv::Point(0, 20), 0.7, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

    for (auto& track : track_list) {
        if (track.GetDetectedCount() < 2) continue;
        const auto& bbox = track.GetLatestData().bbox;
        const auto& object_point = track.GetLatestData().object_point;
        /* Use white rectangle for the object which was not detected but just predicted */
        cv::Scalar color = bbox.score == 0 ? CommonHelper::CreateCvColor(255, 255, 255) : GetColorForId(track.GetId());
        
        
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);
        // cv::resize(depth_result.mat_out, depth_result.mat_out,cv::Size(crop_w, crop_h));
        // cv::rectangle(depth_result.mat_out, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);

        char text[32];
        snprintf(text, sizeof(text), "%.1f,%.1f", object_point.x, object_point.z);
        CommonHelper::DrawText(mat, text, cv::Point(bbox.x, bbox.y - 20), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));


        // 
       if ((bbox.x +(bbox.w / 2.0f) ) < min_x ||(bbox.x +( bbox.w / 2.0f)) > max_x) continue;    
        double sum = 0.0;
        int count = 0;

        //cv::Rect rect(bbox.x, bbox.y, bbox.w, bbox.h);
        //depth_clone(rect).forEach<double>([&](double& pix, const int* pos) {
        //    sum += pix;
        //    count++;
        //});
        
        roi=cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h);
            if(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= crop_w && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= crop_h ){
                       cv::Scalar avg = cv::mean(depth_clone(roi));
                       float depth_avg = avg.val[0];
                       if (depth_avg>0){
                            char depth_text[32];
                            snprintf(depth_text, sizeof(depth_text), "%.1f",depth_avg);
                         CommonHelper::DrawText(mat, depth_text, cv::Point(bbox.x, bbox.y - 40), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));
                       }

                        if (depth_avg >kDepthThreshold)
                        {
                            Object_flag=true;
                            break;
                        }
            }



        //auto& track_history = track.GetDataHistory();
        //for (int32_t i = 1; i < static_cast<int32_t>(track_history.size()); i++) {
        //    cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
        //    cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
        //    cv::line(mat, p0, p1, GetColorForId(track.GetId()));
        //}
    }

    /* Draw on TopView*/
    for (int32_t i = 0; i < static_cast<int32_t>(track_list.size()); i++) {
        auto& track = track_list[i];
        const auto& bbox = track.GetLatestData().bbox;
        const auto& object_point = track.GetLatestData().object_point;
        const auto& topview_point = track.GetLatestData().topview_point;
        cv::Point p = cv::Point(topview_point.x, topview_point.y);

        if (track.GetDetectedCount() < 2) continue;
        cv::circle(mat_topview, p, 3, GetColorForId(track.GetId()), -1);
        cv::circle(mat_topview, p, 3, cv::Scalar(0, 0, 0), 2);

        char text[32];
        snprintf(text, sizeof(text), "%s", bbox.label.c_str());
        CommonHelper::DrawText(mat_topview, text, p, 0.3, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(255, 255, 255), false);
        snprintf(text, sizeof(text), "%.1f,%.1f", object_point.x, object_point.z);
        CommonHelper::DrawText(mat_topview, text, p + cv::Point(0, 15), 0.3, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(255, 255, 255), false);

        //auto& track_history = track.GetDataHistory();
        //for (int32_t i = 1; i < static_cast<int32_t>(track_history.size()); i++) {
        //    cv::Point p0(track_history[i - 1].topview_point.x, track_history[i - 1].topview_point.y);
        //    cv::Point p1(track_history[i].topview_point.x, track_history[i].topview_point.y);
        //    cv::line(mat_topview, p0, p1, GetColorForId(track.GetId()));
        //}
    }
    return Object_flag;
}

cv::Scalar ObjectDetection::GetColorForId(int32_t id)
{
    static constexpr int32_t kMaxNum = 100;
    static std::vector<cv::Scalar> color_list;
    if (color_list.empty()) {
        std::srand(123);
        for (int32_t i = 0; i < kMaxNum; i++) {
            color_list.push_back(CommonHelper::CreateCvColor(std::rand() % 255, std::rand() % 255, std::rand() % 255));
        }
    }
    return color_list[id % kMaxNum];
}
