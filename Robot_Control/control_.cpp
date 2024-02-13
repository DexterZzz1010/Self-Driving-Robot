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

    double Kp = 0.1; // 比例系数
    double Ki = 0.01; // 积分系数
    double Kd = 0.001;
    static double integral = 0;
    static double last_error = 0;

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

   
    

    namedWindow("Lane Detection", WINDOW_NORMAL); // 创建一个窗口
    resizeWindow("Lane Detection", 640, 480); // 设定窗口大小

     while (ros::ok())
    {
        ros::spinOnce(); 
        
        if (flag == true && !img_raw.empty())
        {
            
            frame = img_raw; // 读取一帧图像

            if (frame.empty()) {
                break;
            }

            cv::Mat srcGray;
            cv::cvtColor(frame,srcGray,CV_RGB2GRAY);
            
            cv::Mat mask;
            cv::inRange(frame, cv::Scalar(0, 254, 254), cv::Scalar(0, 255, 255), mask);

            // 利用霍夫变换提取直线
            vector<Vec2f> lines;
            HoughLines(mask, lines, 1, CV_PI / 180, 100, 0, 0);
            //printf("ok");
            // 计算直线的中心线
            float center_line = 0;
            for (size_t i = 0; i < lines.size(); i++) {
                float rho = lines[i][0];
                float theta = lines[i][1];

                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a));
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a));

                // 绘制直线
                line(frame, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);

                // 计算中心线
                center_line += pt1.x + pt2.x;
            }
            center_line /= (2 * lines.size());

            // 绘制中心线
            Point p1(center_line, 0);
            Point p2(center_line, frame.rows);
            line(frame, p1, p2, Scalar(0, 255, 0), 2, LINE_AA);

            double error = center_line - frame.cols / 2;
            double derivative = error - last_error;
            integral += error;
            double output = Kp * error + Ki * integral + Kd * derivative;
            last_error = error;

            // 限制输出在[-1, 1]之间
            if (output > 1) {
                output = 1;
            } else if (output < -1) {
                output = -1;
            }

            // 将输出映射到转向角
            float angle = static_cast<int>(output * 45);
            printf("Angle:               %.2f\n", angle);

            // 控制小车转向
            // 此处需要使用你的具体控制方法
            // ...

            if(angle != 0){
                
                angle_control_data .data= angle;
                angle_control_pub.publish(angle_control_data); 
                rate.sleep();

            }
            


            // 显示结果
            //imshow("Yellow Lane Detection", frame);

        }

    }

    
        
    
    

}
    
