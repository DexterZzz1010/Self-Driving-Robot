#include <opencv2/opencv.hpp> // opencv 库
#include <iostream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>



using namespace std;
using namespace cv;


cv::Mat img_raw;
bool flag=false;

void cameraCallback(const sensor_msgs::ImageConstPtr &msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msg, "bgr8");
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    img_raw = cv_ptrRGB->image; //获取彩色流
    flag = true;
}


int main(int argc, char* argv[])
{

    cv::Mat frame;
    // PID 控制小车运动
    
    // double Kp = 5; // 比例系数
    // double Ki = 0.01; // 积分系数
    // double Kd = 0.001;
    // static double integral = 0;
    // static double last_error = 0;

    /* ROS */
    ros::init(argc, argv, "control");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    //定义一个发布者,将推理结果通过话题/inference_result发送出去
    ros::Publisher angle_control_pub = nh.advertise<std_msgs::Float32>("/angle_control", 1);
    std_msgs::Float32 angle_control_data; //话题传的数据,类型是Float32
	ros::Rate rate(5);//发送频率

    //话题订阅,接收相机发布的彩色流
    image_transport::Subscriber camera_subscriber = it.subscribe("/inference_result", 1, cameraCallback);


    while (ros::ok()){
    
        ros::spinOnce(); 
        if (flag == true && !img_raw.empty())
        {
            
            frame = img_raw; // 读取一帧图像

            if (frame.empty()) {
                continue;
            }



             cv::Mat image;
            cv::cvtColor(frame, image, cv::COLOR_BGR2HSV);


           int center = frame.cols / 2;
            // // 从垂直中线向左查找车道线
            std::vector<cv::Point> left_pts;  // 保存车道线点的容器

            cv::Point left_pt(center, 0);  // 初始化车道线点为中心点
            

            for (int y = 0; y < image.rows; y+=10) {  // 遍历图像每一行

                for (int x = center ; x >= 0; x--) {  // 从中心点向左查找
                    cv::Vec3b hsv_value = image.at<cv::Vec3b>(y, x);
                    int h = hsv_value[0];
                    int s = hsv_value[1];
                    int v = hsv_value[2];
                    if (h >= 20 && h <= 40 && s >= 200 && s <= 255 && v >= 200 && v <= 255) {
                    // 像素点为黄色
                    left_pt = cv::Point2f(x, y); 
                    left_pts.push_back(left_pt); 
                    break;
                    } 
                }
            }

            if(left_pts.empty()){
                continue;
            }


            // 从垂直中线向右查找车道线
            std::vector<cv::Point>right_pts;  // 保存车道线点的容器
            cv::Point right_pt(center, 0);  // 初始化车道线点为中心点
            for (int y = 0; y < image.rows; y+=10) {  // 遍历图像每一行
                for (int x = center + 1; x < image.cols; x++) { // 从中心点向右查找
                   cv::Vec3b hsv_value = image.at<cv::Vec3b>(y, x);
                    int h = hsv_value[0];
                    int s = hsv_value[1];
                    int v = hsv_value[2]; // 获取当前点的 HSV 值
                   if (h >= 20 && h <= 40 && s >= 200 && s <= 255 && v >= 200&& v <= 255) { 
                    // 如果 RGB 值为 (0, 255, 255)，说明找到了车道线
                        right_pt = cv::Point(x, y); // 更新车道线点
                        right_pts.push_back(right_pt); // 保存车道线点
                        break; // 停止查找
                    }
                }
            }

             if(right_pts.empty()){
                continue;
            }

            cv::Vec4f left_line, right_line;
            cv::fitLine(left_pts, left_line, cv::DIST_L2, 0, 0.01, 0.01);  // 拟合左车道线
            cv::fitLine(right_pts, right_line, cv::DIST_L2, 0, 0.01, 0.01);  // 拟合右车道线
            cv::Vec4f center_line=(left_line+right_line)/2;
            
            double err = center_line[2] - center;
            printf("%f",err);
            // double derivative = err- last_error;
            // integral += err;
            // double output = Kp * err+ Ki * integral + Kd * derivative;
            // last_error = err;

            // // 限制输出在[-1, 1]之间
            // if (output > 1) {
            //     output = 1;
            // } else if (output < -1) {
            //     output = -1;
            // }

            // // 将输出映射到转向角
            // float angle = static_cast<int>(output * 45);
            // printf("Angle:               %.2f\n", angle);
            // // 控制小车转向
            // // 此处需要使用你的具体控制方法
            // // ...

            // if(angle != 0){
                
            //     angle_control_data.data = angle;
            //     angle_control_pub.publish(angle_control_data); 
            //     rate.sleep();

            // }
             printf("ok");


            // 显示结果
            //imshow("Lane Detection", frame);

        }

    }

    
        
    
    

}
    
