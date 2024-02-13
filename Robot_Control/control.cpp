#include <opencv2/opencv.hpp> // opencv 库
#include <iostream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>



using namespace std;
using namespace cv;


float err;
bool flag=false;


void errCallback(const std_msgs::Float32 &inference_result)
{
    err = inference_result.data;
    flag=1;
}


int main(int argc, char* argv[])
{

    cv::Mat frame;
    // PID 控制小车运动
    
    double Kp = 0.1; // 比例系数
    double Ki = 0.0; // 积分系数
    double Kd = 0.001;
    static double integral = 0;
    static double last_error = 0;

    /* ROS */
    ros::init(argc, argv, "control");
    ros::NodeHandle nh;
    ros :: Rate loop_rate(1);
    //定义一个发布者,将推理结果通过话题/inference_result发送出去
    ros::Publisher angle_control_pub = nh.advertise<std_msgs::Float32>("/angle_control", 1);
    std_msgs::Float32 angle_control_data; //话题传的数据,类型是Float32
	ros::Rate rate(5);//发送频率

    ros::Subscriber inference_subscriber = nh.subscribe("/inference_result", 1, errCallback);


    while (ros::ok()){
    
        ros::spinOnce(); 

            if(flag!=0){
            double derivative = err- last_error;
            integral += err;
            float output = Kp * err+ Ki * integral + Kd * derivative;
            last_error = err;

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
                
                angle_control_data.data = output;
                angle_control_pub.publish(angle_control_data); 
                rate.sleep();

            }
            //  printf("ok");


            // 显示结果
            //imshow("Lane Detection", frame);

        }

    }

    
        
    
    

}
    
