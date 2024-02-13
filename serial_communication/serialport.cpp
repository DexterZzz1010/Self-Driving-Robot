#include "ros/ros.h"
#include "serial/serial.h"
#include "std_msgs/String.h"
//
#include "std_msgs/Empty.h"

//订阅发来的 速度和角度  消息类型是 geometry_msgs::Twist  所以加上这个头文件
#include "std_msgs/Float32.h"
//数据流头文件  用来做  数字和字符串的相互转换
#include <sstream>
#include "string.h"
using namespace std;

//定义一个全局的 ser   因为在订阅函数中  还需要用到串口发送数据     
serial::Serial ser;
float rvelocity,sdistance;
// unsigned char send_data_buff[100];
bool flag1 = 0;
/*
函数名称：init_seial 
函数参数：无
函数返回值：无
函数功能：初始化串口
函数注意点：注意不同硬件的串口名字可能不同  
*/
void init_serial()
{
      try
      {
            //设置窗口属性 并打开串口   这里的串口名字  不固定  可以百度通过命令语句查询
            ser.setPort("/dev/ttyUSB0");
            //设置波特率
            ser.setBaudrate(115200);
            // 时间设定
            serial::Timeout to = serial::Timeout::simpleTimeout(1000);
            ser.setTimeout(to);
            ser.open();
      }
      catch(serial::IOException& e)
      {
            ROS_ERROR_STREAM("Unable to  open port");
            //return -1;
      }

     //检测串口是否的打开 
      if(ser.isOpen())
      {
            ROS_INFO_STREAM("Serial Port open successfuly!!");
      }
      else
      {
            ROS_ERROR_STREAM("Serial Port open Fault!!");
      }
}
/*
函数名字：read_serialport
函数参数：无
函数返回值：无
函数功能：读取串口缓冲区的内容 并打印到屏幕
函数注意点：无
*/
//void read_serialport()
//{
//      ROS_INFO_STREAM("begin read");
//       if(ser.available())//available
//       {
//            ROS_INFO_STREAM("Reading from serial port");
//            std::string  result ;
//            size_t num2;
//            num2= ser.read(result,ser.available());
//            ROS_INFO_STREAM("the num2 is :"<<num2);
//            ROS_INFO_STREAM("Read:"<<result);
//            ser.flush();
//            ROS_INFO_STREAM("\n");
//       }
//}
void write_serialport(const float robot_velocity, const float seed_distance)
{
      //将接收到速度和角度打印出来
      ROS_INFO_STREAM("moment velocity of robot: "<<robot_velocity);
      ROS_INFO_STREAM("moment distance to seeds:"<<seed_distance);

      uint16_t vel = robot_velocity * 1000;
      uint16_t dis = seed_distance * 1000;

      // static uint16_t vel,dis;
      // vel ++;
      // dis++;

      uint8_t send_arr[8] = {0x55,0,0,0,0,0xaa};
      send_arr[1] = (uint8_t)vel ;
      send_arr[2] = (uint8_t)(vel >> 8) ;
      send_arr[3] = (uint8_t)dis ;
      send_arr[4] = (uint8_t)(dis >> 8) ;

      //通过串口发送给32
      // cout<<"r:"<<result <<endl;
      ser.write(send_arr,6);
      // cout << "robot_velocity = " << robot_velocity << endl;
      // cout << "seed_distance = " << seed_distance << endl;
      // for(auto it : send_arr){
      //       cout << dec << send_arr[0] << ",";
      // }
      // cout << endl;
      // ROS_INFO("datas have been sent!! ");
      // ss.clear();

}
void veloCallBack(const std_msgs::Float32 & vel_msg)
{
    rvelocity = vel_msg.data;
    flag1 = 1;
}

void disCallBack(const std_msgs::Float32 & dis_msg)
{
    sdistance = dis_msg.data;
    flag1 = 1;
}
int main(int argc,char **argv)
{
//       //初始化节点   节点名字： “serial_comm”
      ros::init(argc,argv,"serial_comm");

//       //创建节点句柄
      ros::NodeHandle n;
      //初始化串口
      init_serial();
      ros::Rate loop_rate(1);

      ros::Subscriber velocity_recive = n.subscribe("/velocity_info",10,veloCallBack);
      ros::Subscriber distance_recive = n.subscribe("/seed_distance_info",10,disCallBack);
    ROS_INFO("waiting for info....");
    
      while (ros::ok())
        {
            ros::spinOnce();
            if(rvelocity && sdistance && flag1  != 0)
            {
                  write_serialport(rvelocity,sdistance);
                  flag1 = 0;
            }
            loop_rate.sleep();

        }
}
