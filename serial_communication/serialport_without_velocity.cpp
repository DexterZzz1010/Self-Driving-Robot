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
float angle;
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
            // ls -l /dev/ttyTHS*
            //  ls -l /dev/ttyUSB*
            // sudo chmod 777 /dev/ttyUSB0
            ser.setPort("/dev/ttyUSB0");
            // ser.setPort(" /dev/ttyTHS0");
            
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
void write_serialport( float velx,float rotvelz)
{
      //将接收到速度和角度打印出来
      ROS_INFO_STREAM("moment angle:"<<angle);
      //定义两个字符串  用来存放 转换后的线速度 和角速度 以便  串口发送 
      std::string s_v,s_a;
      

      //定义流 ss_va
      std::stringstream ss_v,ss_a;

      ss_v << 's' << velx << 'v' <<rotvelz<<'a' ;  //将x方向的角速度  转换成流ss_va
      ss_v >> s_v;//将x方向的角速度流转换为字符串

      ROS_INFO_STREAM("v_liner.x = "<<s_v);
      //将转换后的线速度 和 角速度 通过串口 发送给32
      ser.write(s_v);

}

void angleCallBack(const std_msgs::Float32 & angle_msg)
{
    angle = angle_msg.data;
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

      ros::Subscriber distance_recive = n.subscribe("/angle_control",10,angleCallBack);
    ROS_INFO("waiting for info....");
    
      while (ros::ok())
        {
            ros::spinOnce();
            if( flag1  != 0)
            {
                  write_serialport(0,angle);
                  catch (serial::IOException& e){
                        ROS_ERROR_STREAM("Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败，打印错误信息
                  }
                  flag1 = 0;
            }
            loop_rate.sleep();

        }
}
