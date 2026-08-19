#
# mini_pupper_rl_lt/enjoy_pupper.py
#
# how to run
# 1.Gazebo 起動
#  $ source ~/setup-ros2-build
#  $ __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch mini_pupper_simulation bringup.launch.py launch_twist_converter:=False
#
# 2.
#  $ source ~/setup-ros2-mujoco-run
#  $ python enjoy_pupper.py
#
# 3. telop_keyboard
#  $ source ~/setup-ros2-build
#  Twist
#  $ ros2 run teleop_twist_keyboard teleop_twist_keyboard
#
#  TwistStamped
#  $ ros2 run turtlebot3_teleop teleop_keyboard
#
#

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from stable_baselines3 import PPO

from ros_interface import MiniPupperROSInterface, MAX_LIN_X , MAX_LIN_Y , MAX_ANG_Z, MAX_JOINT_RAD, MAX_JOINT_RAD20
from rclpy.executors import MultiThreadedExecutor # ⭕ インポートを追加
import subprocess
import time
import threading
import time

from collections import deque

MAX_ACTION_RAD = 0.5

class enjoyPupperNode(Node):
    def __init__(self, model_path):
        super().__init__('enjoy_pupper_node')
        
        # 1. 学習済みモデル（.zip）の読み込み
        print(f"学習済みモデル {model_path} を読み込み中...")
        self.model = PPO.load(model_path)
        
        # 2. おんちゃん作の ROS 2 インターフェースを起動
        self.ros = MiniPupperROSInterface(twist_stamp=True)

        # --- 【追加】時系列（履歴）の設定 ---
        self.history_len = 6
        self.raw_obs_dim = 31  # cmd_vel(3) + joint_pos(12) + joint_vel(12) + quat(4)

        # 【追加】過去6ステップ分のデータを自動管理するリングバッファ
        self.obs_history = deque(maxlen=self.history_len)

        self.obs_history.clear()
        self.obs_history_init=False

        # 3. 人間からのキーボード操作（/cmd_vel）を待ち受ける
        #self.cmd_vel_sub = self.create_subscription(
        #    Twist,
        #    '/cmd_vel',
        #    self.cmd_vel_callback,
        #    10
        #)
        
        # 4. 制御ループ（100Hz ＝ 10ms周期でAIに計算させる）
        #self.timer = self.create_timer(0.01, self.control_loop)

         # ⭕【修正】制御ループを完全に別スレッドとして立ち上げる
        self.control_thread = threading.Thread(target=self.control_loop)
        # メインプログラムが終了したときにスレッドも一緒に自動終了させる設定
        self.control_thread.daemon = True 
        self.control_thread.start()

        # 綺麗に立った初期姿勢
        self.stand_pose = np.array([0.0]*12, dtype=np.float32) # all 0.0

        self.stand_pose_2 = np.array([
            0.0, 0.0, 0.0, 0.0,
            0.3, -0.6, 0.3, -0.6,
            0.3, -0.6, 0.3, -0.6
        ], dtype=np.float32)

        self.stand_pose_3 = np.array([
            0.0071333,
            0.0071333,
            -0.0071333,
            -0.0071333,
            0.9942296,
            -1.7676911,
            0.9942296,
            -1.7676911,
            0.9942296,
            -1.7676911,
            0.9942296,
            -1.7676911
        ], dtype=np.float32)

        self.init_gz()

    def init_gz(self):

        # 3. 立った状態にする。
        self.ros.send_action(self.stand_pose)
        # 4. 新しい座標トピックが1回届くのを待つために、スピンを多めに回してあげる
        for _ in range(5):
            #rclpy.spin_once(self.ros, timeout_sec=0.01)
            time.sleep(0.01)

        # 1. Gazeboのリセット処理（おんちゃんの既存のコード）
        # robot初期姿勢
        result =  subprocess.run([
            "gz",
            "service",
            "-s",
            "/world/default/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "1000",
            "--req",
            'name:"mini_pupper_2" position:{x:0 y:0 z:0.15} orientation:{w:1}'
        ], capture_output=True, text=True
        )        
        if result.returncode != 0:
            # 終了コードの取得（0は正常終了、それ以外はエラー）
            print("Return code:", result.returncode)

        #rclpy.spin_once(self.ros, timeout_sec=0.01)
        #rclpy.spin_once(self.ros, timeout_sec=0.1)
        # 試験
        if False:
            print("--- 立った姿勢の維持テストを開始します ---")
            while True:
                #time.sleep(0.2)   # 200～500ms程度
                # 命令を維持するために定期的に送る（環境によっては必要）
                self.ros.send_action(self.stand_pose)
                
                # ROS 2の通信を処理する（最重要！）
                #rclpy.spin_once(self.ros, timeout_sec=0.01)
                time.sleep(0.02) # 50Hz周期程度でループ
        if False:
            time.sleep(0.2)   # 200～500ms程度
            self.ros.send_action(self.stand_pose)
            # ROS 2の通信を処理する（最重要！）
            #rclpy.spin_once(self.ros, timeout_sec=0.01)
            time.sleep(0.2) # 50Hz周期程度でループ

    #def cmd_vel_callback(self, msg):
    #    # キーボードから届いた指示速度を、AIに教える用に配列にする
    #    # self.ros.cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)
    #    # ※おんちゃんのノードの変数構造に合わせて設定してください
    #    self.ros.cmd_vel[0] = msg.linear.x
    #    self.ros.cmd_vel[1] = msg.linear.y
    #    self.ros.cmd_vel[2] = msg.angular.z

    def control_loop(self):
        print("🤖 AI推論用制御スレッドが起動しました。")
        WAITE_STEP=2

        # 1. 独立スレッドから、Gazebo側で新しい /joint_states が届くのを【確実に】待つ
        # (メインのExecutorは完全に自由なので、joint_callbackが即座に割り込んでデータを更新できます)
        self.ros.wait_for_gazebo_steps(target_steps=WAITE_STEP,call_th=True)

        while rclpy.ok():

            #print(F"control_loop():#1")
            # 2. 現在のセンサーデータ（Observation）を取得
            obs = self.ros.get_observation()

            #print(F"obs[0]:{obs[0]} ,obs[1]:{obs[1]} ,obs[2]:{obs[2]} ")
            #print(F"cmd_vel:{obs[0]:.2f}, joint[0]:{obs[3]:.2f}, joint[1]:{obs[4]:.2f}")

            if not self.obs_history_init:
                for _ in range(self.history_len):
                    self.obs_history.append(obs)
                self.obs_history_init=True
            else:
                self.obs_history.append(obs) # 自動で一番古いデータが消え、最新が右端に入ります

            # 3. 🧠 学習済みの脳みそに「次の動き（action）」を推論させる！
            # deterministic=True にすることで、ランダムなバタつきを排除した綺麗な動きになります
            #action, _states = self.model.predict(obs, deterministic=True)
            #action, _states = self.model.predict(obs, deterministic=False)
            #action, _states = self.model.predict(self.obs_history, deterministic=True)
            action, _states = self.model.predict(self.obs_history, deterministic=False)

            # 4. 数学的に正しいあの抑制式でモーター角度に変換
            #action = action * 0.2 + home_position
            #MAX_ACTION_RAD = 0.2
            #MAX_ACTION_RAD = 0.5
            #target_angles = action * 0.3 + self.stand_pose
            #target_angles = action * MAX_ACTION_RAD + self.stand_pose
            # 90[度] rad での、逆ノーマライズを、入れる。  changed by nishi 2026.7.28

            target_angles = action * MAX_JOINT_RAD * MAX_ACTION_RAD + self.stand_pose

            # 動きを、 -90度 から +90度 に制限する
            target_angles = np.clip(target_angles, -MAX_JOINT_RAD, MAX_JOINT_RAD)

            # -20度 から +20度 の部分の補正
            # 0, 3, 6, 9 番目のインデックス（ヒップ軸）だけ20度ベースで一括上書き
            target_angles[[0, 3, 6, 9]] = action[[0, 3, 6, 9]] * MAX_JOINT_RAD20 * MAX_ACTION_RAD + self.stand_pose[[0, 3, 6, 9]]
            # 動きを、 -20度 から +20度 に制限する
            target_angles[[0, 3, 6, 9]] = np.clip(target_angles[[0, 3, 6, 9]], -MAX_JOINT_RAD20, MAX_JOINT_RAD20)

            # 5. Gazeboのロボットへ送信！
            self.ros.send_action(target_angles)

            # 1. 独立スレッドから、Gazebo側で新しい /joint_states が届くのを【確実に】待つ
            # (メインのExecutorは完全に自由なので、joint_callbackが即座に割り込んでデータを更新できます)
            self.ros.wait_for_gazebo_steps(target_steps=WAITE_STEP,call_th=True)

            # デバッグ用ログ：交互に綺麗に回っているか確認
            print(f"cmd_vel:{obs[0]:.2f} {obs[2]:.2f}, act lf2:{target_angles[1]:.2f} lf3:{target_angles[2]:.2f} ,rf2:{target_angles[4]:.2f} rf3:{target_angles[5]:.2f}")

            #print(F"call ros.send_action()")

            # ROS 2の通信を更新
            #rclpy.spin_once(self.ros, timeout_sec=0.0)

            #self.ros.wait_for_gazebo_steps(target_steps=2)
            #print(F"call ros.send_action() end!!")

def main():
    rclpy.init()

    # 1. 2つのノードを実体化
    # 保存したモデルのファイル名を指定してください
    enjoy_node = enjoyPupperNode(model_path="outs-05/ppo_minipupper_test_latest.zip")
    ros_interface_node = enjoy_node.ros # 内部のROSノードを取り出す

    # 2. 並行処理できる特別な管理人（Executor）を用意
    executor = MultiThreadedExecutor()

    # 3. 管理人に2つのノードを同時に登録する
    executor.add_node(enjoy_node)
    executor.add_node(ros_interface_node)

    print("🚀 ラジコン操作スタンバイOK！teleop_keyboardを起動して動かしてください。")
    try:
        # 4. まとめてスピン開始！
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        enjoy_node.destroy_node()
        ros_interface_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
