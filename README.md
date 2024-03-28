In the current societal context, mobile robots have been applied in various fields such as exploring hazardous environments, transporting goods, and providing outdoor cleaning services. The research on mobile robots holds significant practical application value.

This thesis focuses on the outdoor scene perception of mobile robots using four deep learning models. This thesis construct a deep learning environment and investigate issues related to model training and optimization, deployment on embedded systems, and robot motion control. The following achievements have been made:

(1) Design and optimization of visual perception models. 

Four deep learning models are employed to achieve four functionalities: object detection, lane detection, scene semantic segmentation, and depth estimation. By adjusting model parameters, optimizing datasets, and refining model structures, this thesis address issues related to errors and accuracy. Post-processing techniques are employed to enable dynamic object tracking and lane fitting.

(2) Deployment of deep learning models for embedded systems. 

This thesis implement the deployment of deep learning models using ONNX-Runtime and TenorRT. Parallel structures are employed for multi-threaded model inference, and visualization is achieved. Through testing the model accuracy and speed, this thesis validate that this research significantly improves the inference speed while maintaining model accuracy.

(3) Extraction of robot navigation parameters and motion control. 

This thesis utilize perspective transformation to convert the front view to a top-down view and propose a method to extract target depth values by combining object detection and depth estimation models. By processing the predictions of the lane detection and scene semantic segmentation models, this thesis obtain the fitted line in the middle of the road. This line serves as the navigation reference, and robot lane following is achieved through PID control.

The achievements of this thesis not only improve the perception and decision-making capabilities of the robot system, meet the requirements for accuracy and stability in practical application scenarios, but also use multi-threading technology to improve inference efficiency and speed. These achievements are expected to improve the safety and reliability of robot systems and meet the demand for intelligent navigation and control in practical application scenarios.
