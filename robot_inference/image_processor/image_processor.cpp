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
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "camera_model.h"
#include "bounding_box.h"
#include "detection_engine.h"
#include "tracker.h"
#include "lane_detection.h"
#include "depth_engine.h"

#include "image_processor_if.h"
#include "image_processor.h"


/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

#define COLOR_BG  CommonHelper::CreateCvColor(70, 70, 70)
static constexpr float kTopViewSizeRatio = 1.0f;


/*** Global variable ***/
bool object_flag = false;

/*** Function ***/
ImageProcessor::ImageProcessor()
{
    frame_cnt_ = 0;
    vanishment_y_ = 1280 / 2;
}

ImageProcessor::~ImageProcessor()
{
}

int32_t ImageProcessor::Initialize(const ImageProcessorIf::InputParam& input_param)
{
    if (object_detection_.Initialize(input_param.work_dir, input_param.num_threads) != ObjectDetection::kRetOk) {
        object_detection_.Finalize();
        return kRetErr;
    }

    if (lane_detection_.Initialize(input_param.work_dir, input_param.num_threads) != LaneDetection::kRetOk) {
        lane_detection_.Finalize();
        return kRetErr;
    }

    if (segmentation_engine_.Initialize(input_param.work_dir, input_param.num_threads) != SemanticSegmentationEngine::kRetOk) {
        segmentation_engine_.Finalize();
        return kRetErr;
    }

    if (depth_engine_.Initialize(input_param.work_dir, input_param.num_threads) != DepthEngine::kRetOk) {
        depth_engine_.Finalize();
        return kRetErr;
    }

    frame_cnt_ = 0; // frame_cnt即推理图片的编号
    vanishment_y_ = 1280 / 2;

    return kRetOk;
}

int32_t ImageProcessor::Finalize(void)
{
    if (object_detection_.Finalize() != ObjectDetection::kRetOk) {
        return kRetErr;
    }

    if (lane_detection_.Finalize() != LaneDetection::kRetOk) {
        return kRetErr;
    }

    if (segmentation_engine_.Finalize() != SemanticSegmentationEngine::kRetOk) {
        return kRetErr;
    }

    if (depth_engine_.Finalize() != DepthEngine::kRetOk) {
        return kRetErr;
    }

    return kRetOk;
}

int32_t ImageProcessor::Command(int32_t cmd)
{
    switch (cmd) {
    case 0:
    default:
        PRINT_E("command(%d) is not supported\n", cmd);
        return kRetErr;
    }
    return kRetOk;
}

int32_t ImageProcessor::Process(const cv::Mat& mat_original, ImageProcessorIf::Result& result)
{
    /*** Initialize internal parameters using input image information ***/
    if (frame_cnt_ == 0) {
        ResetCamera(mat_original.cols, mat_original.rows);
    }

    /*** Run inference ***/
    if (object_detection_.Process(mat_original, mat_transform_, camera_real_) != ObjectDetection::kRetOk) {
        return kRetErr;
    }

    if (lane_detection_.Process(mat_original, mat_transform_, camera_real_) != LaneDetection::kRetOk) {
        return kRetErr;
    }

    SemanticSegmentationEngine::Result segmentation_result;
    if (segmentation_engine_.Process(mat_original, segmentation_result) != SemanticSegmentationEngine::kRetOk) {
        return kRetErr;
    }

    DepthEngine::Result depth_result;
    if (depth_engine_.Process(mat_original, depth_result) != DepthEngine::kRetOk) {
        return -1;
    }

    /*** Create Mat for output ***/
    // 创建图像窗口，用于可视化
    cv::Mat mat = mat_original.clone();
    cv::Mat mat_topview;
    cv::Mat mat_depth;
    cv::Mat mat_segmentation;
    //CreateTopViewMat(mat_original, mat_topview);

    /*** Draw result ***/
    // 绘制输出图 使用engine中的draw函数
    float err_seg=0;
    const auto& time_draw0 = std::chrono::steady_clock::now();
    if (!segmentation_result.image_combined.empty() || !segmentation_result.image_list.empty()) {
        // 绘制语义分割图
        err_seg =DrawSegmentation(mat_segmentation, segmentation_result);
        cv::resize(mat_segmentation, mat_segmentation, mat.size());
        //cv::add(mat_segmentation, mat, mat);
        
        //绘制基于语义分割图的鸟瞰图
        err_seg= CreateTopViewMat(mat_segmentation, mat_topview);
        mat_segmentation = mat_segmentation(cv::Rect(0, vanishment_y_, mat_segmentation.cols, mat_segmentation.rows - vanishment_y_));
    } else {
        mat_topview = cv::Mat::zeros(mat.size(), CV_8UC3);
    }
    
    // 绘制车道线和目标检测
    cv::line(mat, cv::Point(0, camera_real_.EstimateVanishmentY()), cv::Point(mat.cols, camera_real_.EstimateVanishmentY()), cv::Scalar(0, 0, 0), 1);
    float err_lane = lane_detection_.Draw(mat, mat_topview, camera_top_); //车道线
    object_flag=object_detection_.Draw(mat,depth_result, mat_topview); 

    if (!depth_result.mat_out.empty()) {
        // 绘制深度图
        DrawDepth(mat_depth, depth_result);
    }
    const auto& time_draw1 = std::chrono::steady_clock::now();

    
    
    /*** Draw statistics ***/
    // 绘制参数 （time、FPS）
    
    // 绘图时间
    double time_draw = (time_draw1 - time_draw0).count() / 1000000.0;
    //推理时间
    double time_inference = object_detection_.GetTimeInference() + lane_detection_.GetTimeInference() + segmentation_result.time_inference + depth_result.time_inference;
    //FPS
    DrawFps(mat, time_inference, time_draw, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

	mat_topview = mat_topview(cv::Rect(static_cast<int32_t>(mat_topview.cols * 0.2), static_cast<int32_t>(mat_topview.rows * 0.52),
    static_cast<int32_t>(mat_topview.cols * (1.0-2*0.2)),static_cast<int32_t>(mat_topview.rows * (1.0 - 0.52))));
 	cv::resize(mat_topview, mat_topview,cv::Size(mat.cols , mat.rows));	



    /*** Update internal status ***/
    frame_cnt_++; //更新状态（下一张图）


    /*** Return the results ***/
    result.mat_output = mat;
    result.mat_output_segmentation = mat_segmentation;
    result.mat_output_depth = mat_depth;
    result.mat_output_topview = mat_topview;
    result.err=0.1*err_seg+err_lane;
    result.time_pre_process = object_detection_.GetTimePreProcess() + lane_detection_.GetTimePreProcess() + segmentation_result.time_pre_process + depth_result.time_pre_process;
    result.time_inference = object_detection_.GetTimeInference() + lane_detection_.GetTimeInference() + segmentation_result.time_inference + depth_result.time_inference;
    result.time_post_process = object_detection_.GetTimePostProcess() + lane_detection_.GetTimePostProcess() + segmentation_result.time_post_process + depth_result.time_post_process;

    return kRetOk;
}


// 绘制深度图
void ImageProcessor::DrawDepth(cv::Mat& mat, const DepthEngine::Result& depth_result)
{
    if (!depth_result.mat_out.empty()) {
        cv::applyColorMap(depth_result.mat_out, mat, cv::COLORMAP_JET);
    }
}

float ImageProcessor::DrawSegmentation(cv::Mat& mat_segmentation, const SemanticSegmentationEngine::Result& segmentation_result)
{
    /* Draw on NormalView */
    float err = 0;
    if (!segmentation_result.image_combined.empty()) {
        mat_segmentation = segmentation_result.image_combined;
    } else {
        std::vector<cv::Mat> mat_segmentation_list(4, cv::Mat());
#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(segmentation_result.image_list.size()); i++) {
            cv::Mat mat_fp32_3;
            cv::cvtColor(segmentation_result.image_list[i], mat_fp32_3, cv::COLOR_GRAY2BGR); /* 1channel -> 3 channel */
            cv::multiply(mat_fp32_3, GetColorForSegmentation(i), mat_fp32_3);
            mat_fp32_3.convertTo(mat_fp32_3, CV_8UC3, 1, 0);
            mat_segmentation_list[i] = mat_fp32_3;
        }

        //#pragma omp parallel for  /* don't use */
        mat_segmentation = cv::Mat::zeros(mat_segmentation_list[0].size(), CV_8UC3);
        for (int32_t i = 0; i < static_cast<int32_t>(mat_segmentation_list.size()); i++) {
            cv::add(mat_segmentation, mat_segmentation_list[i], mat_segmentation);
        }
    }


    // printf("%f",err);
    return err;
}


void ImageProcessor::DrawFps(cv::Mat& mat, double time_inference, double time_draw, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms], Draw: %.1f [ms]", fps, time_inference, time_draw);
    CommonHelper::DrawText(mat, text, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
}


// 定义语义分割各色块的颜色
cv::Scalar ImageProcessor::GetColorForSegmentation(int32_t id)
{
    switch (id) {
    default:
    case 0: /* BG */ // 背景
        return COLOR_BG;
    case 1: /* road */ //路
        return CommonHelper::CreateCvColor(255, 0, 0);
    case 2: /* curbs */ //障碍物
        return CommonHelper::CreateCvColor(0, 255, 0);
    case 3: /* marks */ //标志
        return CommonHelper::CreateCvColor(0, 0, 255);
    }
}

void ImageProcessor::ResetCamera(int32_t width, int32_t height, float fov_deg)
{
    if (width > 0 && height > 0 && fov_deg > 0) {
        camera_real_.SetIntrinsic(width, height, FocalLength(width, fov_deg));
        camera_top_.SetIntrinsic(static_cast<int32_t>(width * kTopViewSizeRatio), static_cast<int32_t>(height * kTopViewSizeRatio), FocalLength(static_cast<int32_t>(width * kTopViewSizeRatio), fov_deg));
    }
    camera_real_.SetExtrinsic(
        { 0.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, -1.5f, 0.0f }, true);   /* tvec (Oc - Ow in world coordinate. X+= Right, Y+ = down, Z+ = far) */
    camera_top_.SetExtrinsic(
        { 90.0f, 0.0f, 0.0f },    /* rvec [deg] */
        { 0.0f, -8.0f, 11.0f }, true);   /* tvec (Oc - Ow in world coordinate. X+= Right, Y+ = down, Z+ = far) */
    CreateTransformMat();
    vanishment_y_ = std::max(0, std::min(height, camera_real_.EstimateVanishmentY()));
}

// 相机参数
void ImageProcessor::GetCameraParameter(float& focal_length, std::array<float, 3>& real_rvec, std::array<float, 3>& real_tvec, std::array<float, 3>& top_rvec, std::array<float, 3>& top_tvec)
{
    focal_length = camera_real_.fx();
    camera_real_.GetExtrinsic(real_rvec, real_tvec);
    camera_top_.GetExtrinsic(top_rvec, top_tvec);
}

void ImageProcessor::SetCameraParameter(float focal_length, const std::array<float, 3>& real_rvec, const std::array<float, 3>& real_tvec, const std::array<float, 3>& top_rvec, const std::array<float, 3>& top_tvec)
{
    camera_real_.fx() = focal_length;
    camera_real_.fy() = focal_length;
    camera_top_.fx() = focal_length * kTopViewSizeRatio;
    camera_top_.fy() = focal_length * kTopViewSizeRatio;
    camera_real_.SetExtrinsic(real_rvec, real_tvec);
    camera_top_.SetExtrinsic(top_rvec, top_tvec);
    CreateTransformMat();
    vanishment_y_ = std::max(0, std::min(camera_real_.height, camera_real_.EstimateVanishmentY()));
}


// 将2D图像转换成俯视图 转换矩阵
void ImageProcessor::CreateTransformMat()
{
    /*** Generate mapping b/w object points (3D: world coordinate) and image points (real camera) */
    std::vector<cv::Point3f> object_point_list = {   /* Target area (possible road area) */
        { -1.0f, 0, 10.0f },
        {  1.0f, 0, 10.0f },
        { -1.0f, 0,  3.0f },
        {  1.0f, 0,  3.0f },
    };
    std::vector<cv::Point2f> image_point_real_list;
    cv::projectPoints(object_point_list, camera_real_.rvec, camera_real_.tvec, camera_real_.K, camera_real_.dist_coeff, image_point_real_list);

    /* Convert to image points (2D) using the top view camera (virtual camera) */
    std::vector<cv::Point2f> image_point_top_list;
    cv::projectPoints(object_point_list, camera_top_.rvec, camera_top_.tvec, camera_top_.K, camera_top_.dist_coeff, image_point_top_list);

    mat_transform_ = cv::getPerspectiveTransform(&image_point_real_list[0], &image_point_top_list[0]);
}

// 创建俯视图
float ImageProcessor::CreateTopViewMat(const cv::Mat& mat_original, cv::Mat& mat_topview)
{
    /* Perspective Transform */   
    mat_topview = cv::Mat(cv::Size(camera_top_.width, camera_top_.height), CV_8UC3, COLOR_BG);
    //cv::warpPerspective(mat_original, mat_topview, mat_transform_, mat_topview.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    cv::warpPerspective(mat_original, mat_topview, mat_transform_, mat_topview.size(), cv::INTER_NEAREST);

    /* Lane */
    cv::Mat image;
    float err = 0;
    cv::cvtColor(mat_topview, image, cv::COLOR_BGR2HSV);

    int center =mat_topview.cols / 2;
    // // 从垂直中线向左查找车道线
    std::vector<cv::Point> left_pts;  // 保存车道线点的容器

    cv::Point left_pt(center, 0);  // 初始化车道线点为中心点
    

    for (int y = 0; y < image.rows; y+=5) {  // 遍历图像每一行

        for (int x = center ; x >= 0; x--) {  // 从中心点向左查找
            cv::Vec3b hsv_value = image.at<cv::Vec3b>(y, x);
            int h = hsv_value[0];
            int s = hsv_value[1];
            int v = hsv_value[2];
            if (h >= 0 && h <= 10 && s >= 43 && s <= 255 && v >= 46 && v <= 255) {
            // if (h >= 0 && h <= 5 && s >= 200 && s <= 255 && v >= 200 && v <= 255) {
            left_pt = cv::Point2f(x, y); 
            left_pts.push_back(left_pt); 
            break;
            } 
        }
    }

    // 从垂直中线向右查找车道线
    std::vector<cv::Point>right_pts;  // 保存车道线点的容器
    cv::Point right_pt(center, 0);  // 初始化车道线点为中心点
    for (int y = 0; y < image.rows; y+=5) {  // 遍历图像每一行
        for (int x = center + 1; x < image.cols; x++) { // 从中心点向右查找
            cv::Vec3b hsv_value = image.at<cv::Vec3b>(y, x);
            int h = hsv_value[0];
            int s = hsv_value[1];
            int v = hsv_value[2]; // 获取当前点的 HSV 值
            if (h >= 0 && h <= 10 && s >= 43 && s <= 255 && v >= 46 && v <= 255) { 
            // if (h >= 0 && h <=5 && s >= 200 && s <= 255 && v >= 200 && v <= 255) { 
            // 如果 RGB 值为 (0, 255, 255)，说明找到了车道线
                right_pt = cv::Point(x, y); // 更新车道线点
                right_pts.push_back(right_pt); // 保存车道线点
                break; // 停止查找
            }
        }
    }

        if(!right_pts.empty()&&!left_pts.empty()){
            cv::Vec4f left_line, right_line;
            cv::fitLine(left_pts, left_line, cv::DIST_L2, 0, 0.01, 0.01);  // 拟合左车道线
            cv::fitLine(right_pts, right_line, cv::DIST_L2, 0, 0.01, 0.01);  // 拟合右车道线
            cv::Vec4f center_line=(left_line+right_line)/2;

        /* Draw */
           cv:: Point point0;
           point0.x=center_line[2];
           point0.y=center_line[3];
           double k =center_line[1]/center_line[0];
            cv:: Point point1;
            point1.x=0;
            point1.y=k*(0-point0.x)+point0.y;
            cv:: Point point2;
            point2.x=mat_topview.rows;
            point2.y=k*(mat_topview.rows-point0.x)+point0.y;            

            cv::line(mat_topview, point1, point2, cv::Scalar(0,255,0), 2);
            
            err = center_line[2] - center;
    
    return err;
    }





#if 1
    /* Display Grid lines */
    
    /*
    static constexpr int32_t kDepthInterval = 5; // 距离 内参
    static constexpr int32_t kHorizontalRange = 10; //水平线位置
    std::vector<cv::Point3f> object_point_list;
    for (float z = 0; z <= 30; z += kDepthInterval) {
        object_point_list.push_back(cv::Point3f(-kHorizontalRange, 0, z));
        object_point_list.push_back(cv::Point3f(kHorizontalRange, 0, z));
    }
    std::vector<cv::Point2f> image_point_list;
    cv::projectPoints(object_point_list, camera_top_.rvec, camera_top_.tvec, camera_top_.K, camera_top_.dist_coeff, image_point_list); //目标点
    for (int32_t i = 0; i < static_cast<int32_t>(image_point_list.size()); i++) {
        if (i % 2 != 0) {
            cv::line(mat_topview, image_point_list[i - 1], image_point_list[i], cv::Scalar(255, 255, 255)); //车道线
        } else {
            CommonHelper::DrawText(mat_topview, std::to_string(i / 2 * kDepthInterval) + "[m]", image_point_list[i], 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(255, 255, 255), false);
        }
    }
    */
    
    
#endif
}
