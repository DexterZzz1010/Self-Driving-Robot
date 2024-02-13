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

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "depth_engine.h"

/*** Macro ***/
#define TAG "DepthEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if defined(ENABLE_TENSORRT)
#define MODEL_TYPE_ONNX
#else
#define MODEL_TYPE_TFLITE
#endif

#if defined(MODEL_TYPE_TFLITE)
#define MODEL_NAME  "ldrn_kitti_resnext101_pretrained_data_grad_192x320.tflite"
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 192, 320, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#elif defined(MODEL_TYPE_ONNX)

#if 1
#define MODEL_NAME  "ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx"
//#define MODEL_NAME  "LDRN_ResNet18_256_512_Dense.onnx"
#define INPUT_DIMS  { 1, 3, 256, 512 }
#else
#define MODEL_NAME  "ldrn_kitti_resnext101_pretrained_data_grad_192x320.onnx"
#define INPUT_DIMS  { 1, 3, 192, 320 }
#endif

#define INPUT_NAME  "input.1"
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "2499" // resnext101
//#define OUTPUT_NAME "1724"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#endif

/*** Function ***/
// 初始化函数： 工作目录，线程数
int32_t DepthEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
#ifndef ENABLE_DEPTH
    return kRetOk;
#endif

    /* Set model information */
    //onnx 路径
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    // 定义输入tensor
    input_tensor_info_list_.clear(); //清除缓存
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);  //Input_1 FP32
    input_tensor_info.tensor_dims = INPUT_DIMS; // onnx 输入size (N，C，H，W)
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.485f; //normalize
    input_tensor_info.normalize.mean[1] = 0.456f;
    input_tensor_info.normalize.mean[2] = 0.406f;
    input_tensor_info.normalize.norm[0] = 0.229f;
    input_tensor_info.normalize.norm[1] = 0.224f;
    input_tensor_info.normalize.norm[2] = 0.225f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    // 定义输出tensor
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE, IS_NCHW));

    /* Create and Initialize Inference Helper */
    //初始化Inference Helper工具

// 如果原模型是tflite
#if defined(MODEL_TYPE_TFLITE)
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));

// 如果是onnx
#elif defined(MODEL_TYPE_ONNX)
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencv));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt)); // 使用tensorrt
#endif

    //如果没有inference_helper
    if (!inference_helper_) {
        return kRetErr;
    }
    //如果该线程的inference_helper不可用
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();//重新设置 inference_helper
        return kRetErr;
    }
    //如果初始化不成功 重新设置
    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    return kRetOk;
}

// 结束
int32_t DepthEngine::Finalize()
{
#ifndef ENABLE_DEPTH
    return kRetOk;
#endif

    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}

// 构建Engine
int32_t DepthEngine::Process(const cv::Mat& original_mat, Result& result)
{
#ifndef ENABLE_DEPTH
    return kRetOk;
#endif

    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    // 数据预处理
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* do resize and color conversion here because some inference engine doesn't support these operations */
    //resize + color convert
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);

    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols; // W
    input_tensor_info.image_info.height = img_src.rows; // H
    input_tensor_info.image_info.channel = img_src.channels(); // C
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = false; // 颜色 不是bgr
    input_tensor_info.image_info.swap_color = false;
    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /*** Inference ***/
    // 推理
    const auto& t_inference0 = std::chrono::steady_clock::now(); //时间戳
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    //得到output_tensor_info_list_ ？
    const auto& t_inference1 = std::chrono::steady_clock::now(); //时间戳

    /*** PostProcess ***/
    // 可视化处理
    const auto& t_post_process0 = std::chrono::steady_clock::now(); //时间戳
    /* Retrieve the result */
    int32_t output_height = output_tensor_info_list_[0].GetHeight();
    int32_t output_width = output_tensor_info_list_[0].GetWidth();
    // int32_t output_channel = 1;
    float* values = output_tensor_info_list_[0].GetDataAsFloat(); //输出的深度图各像素点的数值
    //printf("%f, %f, %f\n", values[0], values[100], values[400]);
    
    // 将得到的长，宽和像素点的值重组成一张图片
    cv::Mat mat_out = cv::Mat(output_height, output_width, CV_32FC1, values);  /* value has no specific range */
    

    //double depth_min, depth_max;
    //cv::minMaxLoc(mat_out, &depth_min, &depth_max);
    //mat_out.convertTo(mat_out, CV_8UC1, 255. / (depth_max - depth_min), (-255. * depth_min) / (depth_max - depth_min));
    //mat_out.convertTo(mat_out, CV_8UC1);

    //对深度图进行后处理
    mat_out.convertTo(mat_out, CV_8UC1, -5, 255);   /* experimentally deterined */  //经验值？？
    cv::Mat mat_out_ori=mat_out.clone();
    //cv::Mat mat_out_ori=cv::resize(mat_out, cv::Size(crop_w, crop_h));//resize
    // cv::resize(mat_out, mat_out_ori,cv::Size(crop_w, crop_h));
    // cv::resize(mat_out, mat_out,cv::Size(640, 450));
    mat_out = mat_out(cv::Rect(0, static_cast<int32_t>(mat_out.rows * 0.18), mat_out.cols, static_cast<int32_t>(mat_out.rows * (1.0 - 0.18))));
    const auto& t_post_process1 = std::chrono::steady_clock::now(); //时间戳

    /* Return the results */
    result.mat_out = mat_out; //输出的图像
    result.mat_out_ori = mat_out_ori;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0; //预处理时间
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0; //推理时间
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;; //可视化（后处理）时间

    return kRetOk;
}

