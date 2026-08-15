#### mini_pupper_rl_lt  

Mini Pupper2 ros  を Ros2 Jazzy Gazebo(Harmonic) で、強化学習する。第2弾  
SB3 PPO Locomotion Transformer  

##### 1. How to Train  

##### 1.1 Start Gazebo Mini Pupper2  

Term1.  
$ source ~/setup-ros2-build  
$ __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch mini_pupper_simulation bringup.launch.py launch_twist_converter:=False  

#### 1.2 train.ipynb  

Tearm2.  
$ source ~/setup-ros2-mujoco-run  
$ cd sb3-ppo-my/mini_pupper_rl_lt  
$ jupyter notebook  

##### 2. How to inference  

teleop_keyboard で、操作してみる。  

##### 2.1 Start Gazebo Mini Pupper2  

Term1.  
$ source ~/setup-ros2-build  
$ __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch mini_pupper_simulation bringup.launch.py launch_twist_converter:=False  

#### 2.2 Start enjoy_pupper.py  

Term2.  
$ source ~/setup-ros2-mujoco-run  
$ cd sb3-ppo-my/mini_pupper_rl_lt  
$ python  enjoy_pupper.py  

#### 3.2 Teleop_keyboard  

Term3.  
$ source ~/setup-ros2-build  

#####  Twist  
$ ros2 run teleop_twist_keyboard teleop_twist_keyboard  
注) Twist を使うときは、  
enjoy_pupper.py を、少し修正して!!  
````
        # 2. おんちゃん作の ROS 2 インターフェースを起動
        self.ros = MiniPupperROSInterface(twist_stamp=False)
````
  
#####  TwistStamped  
$ ros2 run turtlebot3_teleop teleop_keyboard  
  
##### 3. 参照  

[mini_pupper_ros_my](https://github.com/tosa-no-onchan/mini_pupper_ros_my)
