### sb3-ppo-my  
sb3(Stable Baseline3) ppo(Proximal Policy Optimiaztion) で、強化学習と模倣学習の train さんぷる。  

##### 1. PC 環境  

    Ubuntu 24.04  
    ROS2 Jazzy  

#### 2. Virtual_env 作成  

    $ cd
    $ python3 -m virtualenv mujoco_env  
    $ source mujoco_env/bin/activate  
    $ python -m pip install mujoco  
    $ python -m pip install gymnasium[mujoco] stable-baseline3  
    注) おんちゃんは、 torch 2.10.0 を先にインストールした。  
    $ python -m pip install sb3_contrib  
    $ python -m pip install imitation  
    
#### 3. source script 作成  

sb3 opp のコードを実行する時  
$ source ~/setup-ros2-mujoco-run  

~/setup-ros2-mujoco-run の中身  

````
#echo "ros2"
# 1. まずシステム側の ROS 2 環境を読み込む
source /opt/ros/jazzy/setup.bash
source ~/colcon_ws-jazzy/install/local_setup.bash

# 2. 次に virtualenv を起動する
#source ~/lerobot_env/bin/activate
source ~/mujoco_env/bin/activate

# 3. 【重要】仮想環境にシステム（ROS2）のライブラリパスを認識させる
export PYTHONPATH=$PYTHONPATH:/opt/ros/jazzy/lib/python3.12/site-packages
export PYTHONPATH=$PYTHONPATH:~/colcon_ws-jazzy/install/local_setup.bash # 必要に応じて

export ROS_DOMAIN_ID=30 #TURTLEBOT3
#export ROS_DOMAIN_ID=0 # micro-ROS defaults
export TURTLEBOT3_MODEL=waffle
#export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib


#export PYTHONPATH=$PYTHONPATH:/home/nishi/local/git-download/lerobot/src
# 下記は、build の時は、外す。 run の時だけ使う。
#export PYTHONPATH=~/lerobot_env/lib/python3.12/site-packages:$PYTHONPATH
export PYTHONPATH=~/mujoco_env/lib/python3.12/site-packages:$PYTHONPATH

````

ros2 script を実行する時。  
$ source ~/setup-ros2-build  

~/setup-ros2-build の中身  

````
#echo "ros2"
# add for ROS2
# 1. まずシステム側の ROS 2 環境を読み込む
source /opt/ros/jazzy/setup.bash
source ~/colcon_ws-jazzy/install/local_setup.bash

# 2. 次に virtualenv を起動する
#source lerobot_env/bin/activate

# 3. 【重要】仮想環境にシステム（ROS2）のライブラリパスを認識させる
export PYTHONPATH=$PYTHONPATH:/opt/ros/jazzy/lib/python3.12/site-packages
export PYTHONPATH=$PYTHONPATH:~/colcon_ws-jazzy/install/local_setup.bash # 必要に応じて

export ROS_DOMAIN_ID=30 #TURTLEBOT3
#export ROS_DOMAIN_ID=0 # micro-ROS defaults
export TURTLEBOT3_MODEL=waffle
#export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib
# 下記は、build の時は、外す。 run の時だけ使う。
#export PYTHONPATH=~/lerobot_env/lib/python3.12/site-packages:$PYTHONPATH
````
