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
    
#### 3. source script  

$ mv setup-ros2-mujoco-run ~/  
$ mv setup-ros2-build ~/

##### 3.1 sb3 opp のコードを実行する時  

    $ source ~/setup-ros2-mujoco-run  


##### 3.2 gazebo ros2 script を実行する時  

    $ source ~/setup-ros2-build  

