# gazebo_env.py
# Ver-2
import gymnasium as gym

import numpy as np

from ros_interface import MiniPupperROSInterface, MAX_LIN_X , MAX_LIN_Y , MAX_ANG_Z, MAX_JOINT_RAD, MAX_JOINT_RAD20

import subprocess
import rclpy
import time

import math
import json

from collections import deque

#obj fields
CMD_DIM = 3
JOINT_DIM = 12

ROLL_IDX = CMD_DIM + JOINT_DIM      # 15
PITCH_IDX = ROLL_IDX + 1            # 16

# 例：基準姿勢（Nominal Pose）からの最大変化量を 0.5 rad（約28.6度）に制限する場合
MAX_ACTION_RAD = 0.5        # ここが、ベース。あくまで、 Gazebo 上の話!!
#MAX_ACTION_RAD = 0.8
#MAX_ACTION_RAD = 1.0        # 実機と同じ足の速度にするなら、こちら

class MiniPupperEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.ros = MiniPupperROSInterface()

        # --- 【追加】時系列（履歴）の設定 ---
        self.history_len = 6
        self.raw_obs_dim = 31  # cmd_vel(3) + joint_pos(12) + joint_vel(12) + quat(4)

        #
        # Observation
        #
        # 【変更】Observation Space の shape を (6, 31) の2次元に変更
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            #shape=(14,),      # joint12 + roll + pitch
            #shape=(17,),      # cmd_vel(3) + joint12 + roll + pitch
            #shape=(19,),      # cmd_vel(3) + joint12 + quat(4)
            #shape=(31,),       # cmd_vel(3) + joint_pos(12) + joint_vel(12) + quat(4)
            shape=(self.history_len, self.raw_obs_dim),       # cmd_vel(3) + joint_pos(12) + joint_vel(12) + quat(4)
            dtype=np.float32,
        )
        #
        # Action
        #
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        # 【追加】過去6ステップ分のデータを自動管理するリングバッファ
        self.obs_history = deque(maxlen=self.history_len)

        # Mini Pupper2 の、立った状態の pose
        self.stand_pose_not_use = np.array([
            0.0,  0.994, -1.767,  # 左前
            0.0,  0.994, -1.767,  # 右前
            0.0,  0.994, -1.767,  # 左後
            0.0,  0.994, -1.767   # 右後
        ], dtype=np.float32)

        self.stand_pose = np.array([
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

        self.initial_pose = self.stand_pose

        # ⭕【追加】1エピソードの上限を500ステップ（約5秒）に強制設定
        #self._max_episode_steps = 500  # 0.01
        #self._max_episode_steps = 250   # 0.02
        self._max_episode_steps = 300   # 0.02
        #self._max_episode_steps = 125   # 0.04
        #self.bonus=50.0
        self.bonus=0.1     # changed by nishi 2026.8.19
        self.next_bounus_steps = self._max_episode_steps

        # カウンター用の変数を初期化に追加しておく
        self.episode_steps = 0
        # 回転重視の tarin 時は、 True
        self.rotate_emphance = False
        self.test_id=0

        self.prev_action_norm = None
        self.move_penalty=0.0
        self.move_penalty_cur=0.0

        self.reward_pos_av=0.0
        self.reward_yaw_av=0.0

        # 完走が、出始める迄 stage=0   100,000 - 200,000 steps 迄
        # その後、cmd_vel 追随のフェーズでは、 stage=1 や 2 する
        self.stage=2        # reward stage 0/1/2

        self.min_height=0.12
        self.z_sigma = 0.02   # 許容するブレ幅の感度

        # train 初めは、優しい教育
        if self.stage==0:
            #self.min_height=0.08
            #self.z_sigma = 0.02   # 許容するブレ幅の感度
            self.limit_height=0.05
            # 位置追従オヤツ（ぴったり重なれば最大 1.0点、離れるほどゼロに近づく）
            self.reward_pos_c = -2.0
            # 向き追従オヤツ（理想の方向を向いていれば最大 1.0点）
            self.reward_yaw_c = -4.0

        elif self.stage==1:
            #self.min_height=0.085
            self.z_sigma = 0.035   # 許容するブレ幅の感度
            self.limit_height=0.065
            # 50cmズレたら大幅減点（残り13.5%）。動かないとすぐに0点になるため、前進を促せる
            self.reward_pos_c = -8.0
            # 向きのズレにも厳しくし、前進方向を固定させる（約15度ズレたら残り36%まで減点）
            self.reward_yaw_c = -15.0

        else:
            #self.min_height=0.1
            self.z_sigma = 0.03   # 許容するブレ幅の感度
            self.limit_height=0.065

    def reset(self, seed=None, options=None):
        #print(F"MiniPupperEnv::reset() called!")
        # リセット時にカウンターをゼロに戻す
        self.episode_steps = 0
        self.next_bounus_steps = self._max_episode_steps

        self.prev_action_norm = None
        self.move_penalty=0.0
        self.move_penalty_cur=0.0

        self.reward_pos_av=0.0
        self.reward_yaw_av=0.0

        # 既存のシード初期化
        super().reset(seed=seed)

        # 2. 【追加】ワープ直後に古い記憶（last_yawなど）を即座に強制リセット！
        self.ros.reset_internal_states()

        # 同じ姿勢からの開始に少し、変化をもたせる!!
        # 1. 変化させる幅（ノイズの強さ）を決める（例: ±0.05ラジアン ≒ 約±3度）
        noise_range = 0.05
        noise = np.random.uniform(-noise_range, noise_range, size=self.stand_pose.shape)
        self.initial_pose = self.stand_pose + noise

        # 3. 立った状態にする。
        #self.ros.send_action(self.stand_pose)
        self.ros.send_action(self.initial_pose)
        # 4. 新しい座標トピックが1回届くのを待つために、スピンを多めに回してあげる
        for _ in range(5):
            rclpy.spin_once(self.ros, timeout_sec=0.01)
            time.sleep(0.01)

        # 1. Gazeboのリセット処理（おんちゃんの既存のコード）
        # robot初期姿勢
        result = subprocess.run([
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

        #print(F"MiniPupperEnv::reset() :#3")

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
            #self.ros.send_action(self.stand_pose)
            self.ros.send_action(self.initial_pose)
            # ROS 2の通信を処理する（最重要！）
            rclpy.spin_once(self.ros, timeout_sec=0.01)
            time.sleep(0.2) # 50Hz周期程度でループ

        # 今回の 学習 episod の命題を、設定する。
        self.make_test_cmd()

        # ROS 2ノード側にも新しい命令速度を横流しする
        #self.ros.cmd_vel = self.cmd_vel  <-- これは、不要!! 
        # self.ros.publish_cmd_vel(self.cmd_vel) で、self.ros の中で、normalize したものが、obs に設定される

        # 3. 目標cmd_velを設定する
        #print(F"reset() vx:{self.cmd_vel[0]} ,vy:{self.cmd_vel[1]} ,v_yaw:{self.cmd_vel[2]}")

        #self.ros.publish_cmd_vel_old(self.cmd_vel[0], self.cmd_vel[1], self.cmd_vel[2])
        #self.ros.publish_cmd_vel_old(0.1, 0.0, 0.0)
        self.ros.publish_cmd_vel(self.cmd_vel)
        rclpy.spin_once(self.ros, timeout_sec=0.01)

        #obs = self.ros.get_observation()

        # 1. 既存の関数で『今この瞬間』の生データ(31次元)を取得
        raw_obs = self.ros.get_observation()
        
        # 2. 【追加】バッファをクリアし、開始直後は最初のデータで6マス全て埋める
        self.obs_history.clear()
        for _ in range(self.history_len):
            self.obs_history.append(raw_obs)
            
        #return obs, {}
        # 3. [6, 31] の形状の NumPy 配列に変換して返す
        return np.array(self.obs_history, dtype=np.float32),{}

    def make_test_cmd(self):
        #MAX_LIN_X = 0.26  # m/s
        #MAX_LIN_Y = 0.13  # m/s
        #MAX_ANG_Z = 1.82  # rad/s

        # ちょっと、直進だけで、テスト
        #self.test_id=0
        #self.test_id=9

        # MAX_LIN_X = 0.26  # m/s
        # 注) 1steps 20[ms] で、500steps で、 Max 0.38[M] なので、 0.2 [M] ずれたら、おしおきか!!

        if self.test_id==0:
            vx = MAX_LIN_X
            vy=0.0
            v_yaw=0.0
        elif self.test_id==1:
            vx= MAX_LIN_X * -0.5
            vy=0.0
            v_yaw=0.0
        elif self.test_id==2:
            vx=0.0
            vy=0.0
            v_yaw= MAX_ANG_Z
        elif self.test_id==3:
            vx=0.0
            vy=0.0
            v_yaw= MAX_ANG_Z * -1.0
        elif self.test_id==4:
            vx=0.0
            vy=0.0
            v_yaw=0.0
        #
        elif self.test_id==5:
            vx= MAX_LIN_X * 0.5
            vy=0.0
            v_yaw=0.0
        elif self.test_id==6:
            vx= MAX_LIN_X * -0.25
            vy=0.0
            v_yaw=0.0
        elif self.test_id==7:
            vx=0.0
            vy=0.0
            v_yaw=  MAX_ANG_Z *0.5
        elif self.test_id==8:
            vx=0.0
            vy=0.0
            v_yaw= MAX_ANG_Z * -0.5

        else:
            # 回転重視の train
            if self.rotate_emphance:
                # ⭕【大改造】エピソード開始時に、今回の命令をランダムに決定！
                # 前進・バック（-0.1 〜 0.15 m/s）、旋回（-0.6 〜 0.6 rad/s）
                # 前進・バック（-0.1 〜 0.26 m/s）、旋回（-1.82 〜 1.82 rad/s）  <-- こちらにする
                #vx = float(np.random.uniform(-0.1, 0.15))
                vx = float(np.random.uniform(-0.1, MAX_LIN_X))
                vy = 0.0 # 横歩きは最初は0固定が安全です
                #v_yaw = float(np.random.uniform(-0.6, 0.6))
                v_yaw = float(np.random.uniform(MAX_ANG_Z * -1.0, MAX_ANG_Z))
                
                # 10%の確率で「その場にピタッと止まれ」の命令も混ぜる（これも大事な訓練）
                if np.random.rand() < 0.1:
                    vx, v_yaw = 0.0, 0.0
            else:
                # ⭕ 70%の確率は「旋回なし」の純粋な前進・後退特訓にする！
                if np.random.rand() < 0.7:
                    #vx = float(np.random.uniform(-0.1, 0.15))
                    vx = float(np.random.uniform(-0.1, MAX_LIN_X))
                    vy = 0.0
                    v_yaw = 0.0  # 旋回は強制的にゼロ！
                else:
                    # 残りの30%で旋回やその場停止を混ぜる
                    vx = float(np.random.uniform(-0.05, 0.05))
                    vy = 0.0
                    #v_yaw = float(np.random.uniform(-0.6, 0.6))
                    v_yaw = float(np.random.uniform(MAX_ANG_Z * -1.0, MAX_ANG_Z))
            
        # 内部変数に保存（Observationに反映される）
        self.cmd_vel = np.array([vx, vy, v_yaw], dtype=np.float32)
        self.test_id += 1
        #if self.test_id > 12:
        #if self.test_id > 30:
        if self.test_id > 18:
            self.test_id=0
            self.rotate_emphance = not self.rotate_emphance

    def step(self, action):
        # 毎ステップ、カウンターを1増やす
        self.episode_steps += 1
        #print(F"step(): self.episode_steps:{self.episode_steps}")
        #
        # PPO
        #  ↓
        # Float64MultiArray
        #
        # action を制限する
        # 1 step 辺り Gazebo で、 20ms から 40ms くらい待つのが良いそう!!
        WAITE_STEP=2
        #WAITE_STEP=4

        #action = action * 0.2 + home_position
        #action = action * MAX_ACTION_RAD + self.stand_pose
        # 90[度] rad での、逆ノーマライズを、入れる。  changed by nishi 2026.7.28

        stand_pose = self.initial_pose

        action_norm = np.zeros(12)
        action_norm_check = np.zeros(12)

        action_norm = action * MAX_JOINT_RAD * MAX_ACTION_RAD + stand_pose

        USE_PUPPER_JOINT_ANGLE=True
        # どちらを使うか、悩ましいところ!!
        if not USE_PUPPER_JOINT_ANGLE:
            # 下記は、 model の output を 直使う場合
            action_norm_chek = action * MAX_JOINT_RAD
        else:
            # 下記は、 mini pupper の joint 角を使う場合
            action_norm_chek = action * MAX_JOINT_RAD * MAX_ACTION_RAD

        # 動きを、 -90度 から +90度 に制限する
        action_norm = np.clip(action_norm, -MAX_JOINT_RAD, MAX_JOINT_RAD)

        # -20度 から +20度 の部分の補正
        # 0, 3, 6, 9 番目のインデックス（ヒップ軸）だけ20度ベースで一括上書き
        action_norm[[0, 3, 6, 9]] = action[[0, 3, 6, 9]] * MAX_JOINT_RAD20 * MAX_ACTION_RAD + stand_pose[[0, 3, 6, 9]]
        # 動きを、 -20度 から +20度 に制限する
        action_norm[[0, 3, 6, 9]] = np.clip(action_norm[[0, 3, 6, 9]], -MAX_JOINT_RAD20, MAX_JOINT_RAD20)

        if not USE_PUPPER_JOINT_ANGLE:
            action_norm_chek[[0, 3, 6, 9]] = action[[0, 3, 6, 9]] * MAX_JOINT_RAD20
        else:
            action_norm_chek[[0, 3, 6, 9]] = action[[0, 3, 6, 9]] * MAX_JOINT_RAD20 * MAX_ACTION_RAD

        # 初期化：1ステップ目のために予めエラーが出ないよう 0 で初期化しておく
        velocity_penalty = 0.0

        if self.prev_action_norm is not None:
            # --- 【追加】回転角速度ペナルティの計算 ---
            # 1ステップあたりの経過時間(秒)。WAITE_STEP=2 で Gazeboが20ms周期なら 0.02秒
            dt = 0.02 
            
            # 前回の目標角度との差分[rad]から、擬似的な「回転角速度[rad/s]」を計算
            joint_velocities = (action_norm_chek - self.prev_action_norm) / dt

            # 1. 各関節の「絶対角速度」を計算
            abs_velocities = np.abs(joint_velocities)
            abs_velocities_max = np.max(abs_velocities)

            #print(F"abs_velocities_max:{abs_velocities_max}")

            # 2. 許容する基準値（しきい値）を設定
            # 基準値をモデルの生の激しさに合わせて引き上げる
            # 2. 許容する基準値（実機の物理限界 220rpm ≒ 23 rad/s）
            threshold_vel = 20.0
            #threshold_vel = 18.0
            #threshold_vel = 15.0
            #threshold_vel = 10.0
            #threshold_vel = 9.0

            # 3. 基準を超えた超過分だけを抽出（マイナスは0にする）
            excess_velocities = np.maximum(0.0, abs_velocities_max - threshold_vel)

            # 4. 超過分を2乗して平均を取り、ペナルティにする
            # 角速度の二乗和（または絶対値の和）を計算
            # 各関節の角速度が大きいほど、ペナルティが跳ね上がります

            if not USE_PUPPER_JOINT_ANGLE:
                #w_vel = 0.001  # 減点の重み係数（足の動きを見ながら調整）
                w_vel = 0.0001  # 減点の重み係数（足の動きを見ながら調整）
            else:
                w_vel = 0.001  # 減点の重み係数（足の動きを見ながら調整）
                #w_vel = 0.0001  # 減点の重み係数（足の動きを見ながら調整）
                #w_vel = 0.0005  # 減点の重み係数（足の動きを見ながら調整）

            #velocity_penalty = w_vel * np.max(np.square(excess_velocities))
            velocity_penalty = w_vel * np.square(excess_velocities) # 配列ではないので np.max や np.sum は不要

            #if velocity_penalty > 0.0:
            #    print(F"velocity_penalty:{velocity_penalty:.3f} ,abs_velocities_max:{abs_velocities_max:.3f}")

        # 次のステップのために現在の目標角度を保存
        self.prev_action_norm = action_norm_chek.copy()

        self.ros.send_action(action_norm)
        #
        # 1 step 辺り Gazebo で、 20ms から 40ms くらい待つのが良いそう!!
        #
        self.ros.wait_for_gazebo_steps(target_steps=WAITE_STEP)

        #
        # Observation
        #
        obs = self.ros.get_observation()
        #print(F'step():#2 obs[0]:{obs[0]:.2f} ,obs[1]:{obs[1]:.2f} ,obs[2]:{obs[2]:.2f}')

        self.obs_history.append(obs) # 自動で一番古いデータが消え、最新が右端に入ります

        #
        # 転倒、コースズレ判定
        #
        terminated, reward = self.check_fall(obs)   # # コケたり脱線したら即終了
        if terminated:
            # 1. コケたら即座に大減点（お説教）して終了
            #reward = -10.0
            # 【修正】コケた大減点（-10.0）に、その時の足の暴走ペナルティも上乗せして引く！
            #reward = -10.0 - velocity_penalty
            #reward = -50.0 - velocity_penalty
            reward -= velocity_penalty
            #reward = float(-self._max_episode_steps) - velocity_penalty
        else:
            #
            # Reward
            reward = self.compute_reward(obs,action) - velocity_penalty

        #truncated = False
        # ⭕【修正】500ステップに達したら truncated（時間切れ）を True にする！
        truncated = (self.episode_steps >= self._max_episode_steps) 

        # ==================================================================
        # ⭕【追加】500ステップ見事に完走したときの「特大ご褒美ボーナス」
        # ==================================================================
        # コケずに（not terminated）、500ステップ耐え抜いた（truncated）瞬間なら
        #if truncated and not terminated:
        if self.episode_steps >= self.next_bounus_steps and not terminated:
            bonus = 0.0
            self.reward_pos_av /= self._max_episode_steps
            self.reward_yaw_av /= self._max_episode_steps
            if self.stage == 2:
                if self.reward_pos_av > 0.0 and self.reward_yaw_av > 0.0:
                    bonus = self.bonus  # 🎁 特大ボーナス（普段の数十倍の価値のオヤツ！）
            else:
                if self.reward_pos_av > 0.8 and self.reward_yaw_av > 0.8:
                    bonus = self.bonus  # 🎁 特大ボーナス（普段の数十倍の価値のオヤツ！）

            reward += bonus
            self.next_bounus_steps = self.next_bounus_steps + self._max_episode_steps
            print(f"🎉 {self._max_episode_steps}ステップ完走！特大ボーナス {bonus} 点をプレゼント！ reward:{reward:.2f} tilt_penalty:{self.tilt_penalty:.2f}")

        #print(F"steps:{self.episode_steps} reward:{reward:.2f}")

        # ==================================================================

        # 4. バッファを [6, 31] の NumPy 配列に変換
        next_obs = np.array(self.obs_history, dtype=np.float32)

        return (
            #obs,
            next_obs,
            reward,
            terminated,
            truncated,
            {}
        )

    def body_move_check(self):
        # 回転角速度
        current_vyaw = self.ros.get_yaw_velocity()
        # 前進速度
        current_vx =self.ros.get_forward_velocity()
        # 横移動速度
        current_vy =self.ros.get_side_velocity()

        # 1. 実際の胴体の「平面移動スピード」を計算 [m/s]
        actual_linear_speed = np.linalg.norm([
            #self.actual_linear_vel.x,
            current_vx,
            #self.actual_linear_vel.y
            current_vy
        ])
        # 2. 実際の胴体の「旋回（Yaw）スピードの絶対値」を取得 [rad/s]
        # ※ロボットが横転したときのブレ（Roll/Pitch）を除外するため、Z軸（Yaw）の回転だけを見ます
        #actual_angular_speed = abs(self.actual_angular_vel.z)
        actual_angular_speed = abs(current_vyaw)

        # 3. AIから課されている「指令速度」の大きさを計算
        target_linear_speed = np.linalg.norm([self.cmd_vel[0], self.cmd_vel[1]])
        target_angular_speed = abs(self.cmd_vel[2])

        # 4. サボり判定用の閾値（しきい値）を設定
        # 命令が出ている（> 0.05）のに、実際の動きが極小（< 0.03）ならサボりとみなす
        is_linear_lazy = (target_linear_speed > 0.05) and (actual_linear_speed < 0.03)
        is_angular_lazy = (target_angular_speed > 0.05) and (actual_angular_speed < 0.03)

        # 5. 条件判定：移動命令か回転命令のどちらかで「サボり」が発生しているか
        if is_linear_lazy or is_angular_lazy:
            # 【サボり確定】一発レッドカードの特大マイナス
            #penalty = -10.0
            #penalty = -0.5  
            penalty = -1.0  
            #penalty = -0.7
            # print("😑 命令が出ているのに、胴体の移動または回転が完全に止まっています！")
        else:
            penalty = 0.0
        return penalty

    def move_check(self):
        penalty = 0.0
        # cmd_vel は、停止以外
        if np.sum(np.abs(self.cmd_vel)) > 0.0:
            if self.ros.velocities is not None:
                # 1. 全関節速度の「絶対値の平均」を算出する
                mean_velocity = np.mean(np.abs(self.ros.velocities))
                
                # 2. 閾値（例: 0.02 rad/s 以下ならサボりとみなす）
                # ※実機の挙動を見ながら 0.01 〜 0.05 あたりで微調整してください
                #if mean_velocity < 0.02:
                if mean_velocity < 0.05:
                    # 【サボり確定】毎ステップの報酬から引くための大きなマイナス値を返す
                    penalty = -10.0  
                else:
                    # ちゃんと足を動かしているならペナルティはゼロ
                    penalty = 0.0
        return penalty

    def compute_reward(self, obs,action):
        # 1. コケたら即座に大減点（お説教）して終了
        #if self.check_fall(obs):
        #    return -10.0

        # 2. 現実の位置・向き と 仮想の理想位置・向き を取得
        actual_x = self.ros.current_x
        actual_y = self.ros.current_y
        actual_z = self.ros.current_z
        actual_yaw = getattr(self.ros, 'last_yaw', 0.0)
        
        virt_x = self.ros.pupper_virt_odom['x']
        virt_y = self.ros.pupper_virt_odom['y']
        virt_yaw = self.ros.pupper_virt_odom['yaw']

        #print(F"virt_odom: {virt_x}, {virt_y}, {virt_yaw}")

        # 3. 【位置のズレ（エラー）】 理想と現実の2点間の直線距離の二乗
        pos_error_sq = (virt_x - actual_x)**2 + (virt_y - actual_y)**2
        reward_pos=0.0
        if self.stage==0 or self.stage==1:
            # 位置追従オヤツ（ぴったり重なれば最大 1.0点、離れるほどゼロに近づく）
            #reward_pos = np.exp(-2.0 * pos_error_sq)
            reward_pos = np.exp(self.reward_pos_c * pos_error_sq)
            #print(F"pos_error_sq:{pos_error_sq:.3f} reward_pos:{reward_pos:.3f}")
        else:
            # - を入れて、めりはりをつける!!
            # 1. 前提条件と最大許容値の設定
            threshold_pos = 0.5  # 境界線：50cm（0.5m）
            max_pos_error = 3.0  # 想定される最大のズレ：3.0m（目標が3m先のため）

            distance_error = np.sqrt(pos_error_sq) # 直線距離（※np.linalg.normは不要です）

            if distance_error < threshold_pos:
                # 【ボーナス】50cm以内ならプラス（ぴったりで1.0点、50cm離れると0.0点に近づく）
                # 元のexpの形を活かしつつ、50cmでほぼ0（0.01）になるよう係数を-18.0に調整
                reward_pos = np.exp(-18.0 * pos_error_sq)
            else:
                # 【ペナルティ】50cm以上離れたら、0.0 〜 -1.0 に綺麗に収める
                # 50cmを超えて「最大3.0mまで」の間に、どれだけはみ出したかの比率を計算
                reward_pos = - (distance_error - threshold_pos) / (max_pos_error - threshold_pos)
                # 万が一、3.0m以上離れてマイナスが大きくなりすぎないようにガード（-1.0で固定）
                reward_pos = max(reward_pos, -1.0)

            #print(F"episode_steps:{self.episode_steps} distance_error:{distance_error:.3f} reward_pos:{reward_pos:.3f}")

        #print(F"pos_error_sq:{pos_error_sq:.3f} reward_pos:{reward_pos:.3f}")

        # 4. 【向きのズレ（エラー）】 理想と現実の角度の差分
        reward_yaw =0.0
        yaw_error = virt_yaw - actual_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error)) # -π〜+π正規化
        if self.stage==0 or self.stage==1:
            # 向き追従オヤツ（理想の方向を向いていれば最大 1.0点）
            #reward_yaw = np.exp(-4.0 * (yaw_error**2))
            reward_yaw = np.exp(self.reward_yaw_c * (yaw_error**2))
        else:
            # - を入れて、めりはりをつける!!
            # 1. 20度をラジアンに変換（約 0.349 rad）
            threshold_yaw = np.radians(20.0)
            max_yaw = np.radians(180.0)  # 最大のズレ（πラジアン = 180度）
            # 2. 条件分岐で報酬を計算
            if abs(yaw_error) > threshold_yaw:
                # 【21度〜180度を 0.0 〜 -1.0 に収める計算】
                # 20度を超えた分の距離が、残り（20度から180度まで）に対してどれだけの比率か
                reward_yaw = - (abs(yaw_error) - threshold_yaw) / (max_yaw - threshold_yaw)
            else:
                # 20度以内ならプラス（ズレ0で1.0点、20度で0.0点）
                reward_yaw = 1.0 - (abs(yaw_error) / threshold_yaw)

        #print(F"episode_steps:{self.episode_steps} yaw_error:{math.degrees(yaw_error):.1f}[度] reward_yaw:{reward_yaw:.3f}")

        # 5. 【姿勢の綺麗さペナルティ】
        roll = self.ros.roll
        pitch = self.ros.pitch

        # ピッチの角速度（シーソーの激しさ）を取得（なければ0.0）
        pitch_vel = getattr(self.ros, 'pitch_velocity', 0.0)

        # 傾き(角度)のペナルティに加えて、前後の「揺れの激しさ(角速度)」も加算
        # 係数（例: 0.1）は、揺れの強さに応じて調整してください
        tilt_penalty = (roll ** 2) + (pitch ** 2) + 0.1 * (pitch_vel ** 2)
        #tilt_penalty = (roll ** 2) + (pitch ** 2)

        height_penalty = 0.0
        # 6. 高さの報酬
        if self.stage==0:
            # 目標の高さに近いほど 1.0 に近づき、離れると 0 になる報酬
            height_penalty = np.exp(-((actual_z - self.min_height) ** 2) / (self.z_sigma ** 2))
        else:
            threshold_z = self.min_height - self.z_sigma 
            # - を入れて、めりはりをつける!!
            if actual_z > threshold_z:
                # 0.10m から 0.12m の間で 0.0 から 1.0 になる計算
                # ※0.12mを超えて高すぎても減点したい場合は、np.clipで1.0に固定するか調整可能
                height_penalty = (actual_z - threshold_z) / (self.min_height - threshold_z)
                height_penalty = np.clip(height_penalty, 0.0, 1.0) # 高すぎても最大1.0で固定
            else:
                # 0.10m から 0.05m の間で 0.0 から -1.0 になる計算
                height_penalty = - (threshold_z - actual_z) / (threshold_z - self.limit_height)
                height_penalty = np.clip(height_penalty, -1.0, 0.0) # 最低地上高以下は-1.0で固定
                #print(F'compute_reward()  actual_z:{actual_z:.3f} penalty:{height_penalty:.3f}')

        if False:
            #move_penalty = self.move_check()
            move_penalty = self.body_move_check()
            if move_penalty == 0.0:
                self.move_penalty_cur=0.0
                self.move_penalty = 0.0
            else:
                self.move_penalty_cur += move_penalty
            if self.move_penalty_cur < -3.0:
                self.move_penalty = -0.5

        if False:
            print(F'reward_pos:{reward_pos:.3f}')   # こいつが増えていくか?
            print(F'reward_yaw:{reward_yaw:.3f}')
            print(F'height_penalty:{height_penalty:.3f}')
            print(F'tilt_penalty:{tilt_penalty:.3f}')
            print(F'move_penalty:{move_penalty:.3f}')

        # 7. 【最終報酬のドッキング】
        #reward = (0.7 * reward_pos + 0.3 * reward_yaw) - 0.5 * tilt_penalty + height_penalty
        reward = (reward_pos + reward_yaw) - 0.5 * tilt_penalty + height_penalty

        self.tilt_penalty = -0.5 * tilt_penalty

        self.reward_pos_av += reward_pos
        self.reward_yaw_av += reward_yaw

        #生存報酬（Alive Bonus）: 転ばずに1ステップ生き延びるごとに、小さなプラス報酬（例: +1）を与えます。
        # これにより「長く立っていること」自体を学習させます。
        #if self.episode_steps % 5 ==0 and self.move_penalty == 0.0 and height_penalty >= 0.0:
        if self.stage == 2:
            if reward_pos > 0.0 and reward_yaw > 0.0:
                reward += 1
                #reward += 2.0
            else:
                #reward -= 1.0
                pass
        else:
            if reward_pos > 0.8 and reward_yaw > 0.8:
                reward += 1
            else:
                reward -= 0.5

        #print(F'compute_reward():#2 reward:{reward:.3f}')
        return float(reward)

    def compute_reward_old(self, obs,action):
        # 1. コケたら即座に大減点（お説教）して終了
        #if self.check_fall(obs):
        #    return -10.0

        # 2. 現実の位置・向き と 仮想の理想位置・向き を取得
        actual_x = self.ros.current_x
        actual_y = self.ros.current_y
        actual_z = self.ros.current_z
        actual_yaw = getattr(self.ros, 'last_yaw', 0.0)
        
        virt_x = self.ros.pupper_virt_odom['x']
        virt_y = self.ros.pupper_virt_odom['y']
        virt_yaw = self.ros.pupper_virt_odom['yaw']

        #print(F"virt_odom: {virt_x}, {virt_y}, {virt_yaw}")

        # 3. 【位置のズレ（エラー）】 理想と現実の2点間の直線距離の二乗
        pos_error_sq = (virt_x - actual_x)**2 + (virt_y - actual_y)**2
        # 位置追従オヤツ（ぴったり重なれば最大 1.0点、離れるほどゼロに近づく）
        reward_pos = np.exp(-2.0 * pos_error_sq)

        #print(F"pos_error_sq:{pos_error_sq:.3f} reward_pos:{reward_pos:.3f}")

        # 4. 【向きのズレ（エラー）】 理想と現実の角度の差分
        yaw_error = virt_yaw - actual_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error)) # -π〜+π正規化
        # 向き追従オヤツ（理想の方向を向いていれば最大 1.0点）
        reward_yaw = np.exp(-4.0 * (yaw_error**2))

        # 5. 【姿勢の綺麗さペナルティ】
        roll = self.ros.roll
        pitch = self.ros.pitch

        # ピッチの角速度（シーソーの激しさ）を取得（なければ0.0）
        pitch_vel = getattr(self.ros, 'pitch_velocity', 0.0)

        # 傾き(角度)のペナルティに加えて、前後の「揺れの激しさ(角速度)」も加算
        # 係数（例: 0.1）は、揺れの強さに応じて調整してください
        #tilt_penalty = (roll ** 2) + (pitch ** 2) + 0.1 * (pitch_vel ** 2)
        tilt_penalty = (roll ** 2) + (pitch ** 2)

        height_penalty = 0.0
        #min_height=0.07
        #min_height=0.08
        #min_height=0.085
        #min_height=0.087
        #min_height=0.09
        # 6. 高さの報酬
        if actual_z <= self.min_height:
            #height_penalty = (actual_z - min_height) * 100.0
            #height_penalty = (actual_z - min_height) * 150.0 * 2.0
            #height_penalty = -1.5
            height_penalty = -0.5
            #print(F'compute_reward()  actual_z:{actual_z:.3f} penalty:{height_penalty:.3f}')
        else:
            height_penalty = (actual_z - self.min_height) * 10.0 * 2.0
            #print(F'compute_reward()  actual_z:{actual_z:.3f} penalty:{height_penalty:.3f}')

        # 7. 【最終報酬のドッキング】
        # 位置が合っていて（0.7）、かつ向きも合っている（0.3）ときが高得点。
        # そこからブサイク姿勢ペナルティを引き算する
        if self.rotate_emphance:
            reward = (0.7 * reward_pos + 0.3 * reward_yaw) - 0.5 * tilt_penalty + height_penalty
        else:
            # ⭕ 最終報酬のドッキングを「位置（距離）9割」に尖らせる！
            reward = (0.9 * reward_pos + 0.1 * reward_yaw) - 0.5 * tilt_penalty + height_penalty

        #生存報酬（Alive Bonus）: 転ばずに1ステップ生き延びるごとに、小さなプラス報酬（例: +1）を与えます。
        # これにより「長く立っていること」自体を学習させます。
        if self.episode_steps % 5 ==0:
            reward += 1

        #print(F'compute_reward():#2 reward:{reward}')
        return float(reward)

    #---
    # 転倒、致命的 コースズレ判定
    #---
    def check_fall(self, obs):
        # 1. 既存のロール・ピッチのコケ判定
        roll = self.ros.roll
        pitch = self.ros.pitch
        #print(F'check_fall(): roll:{roll} pitch:{pitch}')
        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            #print(f"コケたのでお説教！ roll: {abs(roll):.2f} , pitch:{abs(pitch):.2f}")
            #return True,-120.0
            #return True,-250.0
            return True,-10.0

        # 2. 【進化】仮想の理想位置から 1.0メートル以上 脱線したらお説教リセット！
        actual_x = self.ros.current_x
        actual_y = self.ros.current_y
        actual_z = self.ros.current_z
        virt_x = self.ros.pupper_virt_odom['x']
        virt_y = self.ros.pupper_virt_odom['y']
        
        pos_error = np.sqrt((virt_x - actual_x)**2 + (virt_y - actual_y)**2)

        if False:
            if pos_error > 1.5:
                print(f"理想のルートから脱線（横滑り）したのでお説教！ (誤差: {pos_error:.2f}m)")
                return True,-20.0

        # ③ よそ見判定：【45度(0.78) → 90度(1.57) または【一旦コメントアウト】】
        # 歩き始めは機体が激しく左右に首を振ります。45度制限があると、前進しようと頑張っている最中に即死します。

        # 3. 【進化】理想の向きから 45度以上 よそ見したらお説教リセット！
        actual_yaw = getattr(self.ros, 'last_yaw', 0.0)
        virt_yaw = self.ros.pupper_virt_odom['yaw']
        
        yaw_error = virt_yaw - actual_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # 学習初期は、よそ見でお説教（即死）にするのではなく、
        # compute_rewardの「reward_yaw（減点）」だけでジワジワ教える方が上手くいきます。
        #if abs(yaw_error) > 1.57: # 完全に真後ろや真横を向いたらリセット、くらいにする
        #    print(f"命令された向きからよそ見したのでお説教！ (ズレ: {np.degrees(yaw_error):.1f}度)")
        #    return True
        if True:
            if actual_z < self.limit_height: 
                print(f"伏せしたので、お説教！ (高さ: {actual_z:.3f}M)")
                return True,-20.0

        if False:
            #move_penalty = self.move_check()
            move_penalty = self.body_move_check()
            if move_penalty == 0.0:
                self.move_penalty=0.0
            else:
                self.move_penalty += move_penalty
            #if self.move_penalty < -30.0:
            if self.move_penalty < -20.0:
                print(f"動かないので、お説教！ (点: {self.move_penalty:.3f})")
                return True, self.move_penalty

        return False,0.0
    
