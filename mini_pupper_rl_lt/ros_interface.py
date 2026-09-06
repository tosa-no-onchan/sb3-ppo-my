# ros_interface.py
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu

from std_msgs.msg import Float64MultiArray
import numpy as np

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Twist, TwistStamped
from tf2_msgs.msg import TFMessage

from geometry_msgs.msg import PoseArray # ⭕ PoseArray をインポート

from collections import deque  # 1. ライブラリをインポート

#from tf_transformations import euler_from_quaternion

# /cmd_vel normalize
# 設計した最大値の定義
#MAX_LIN_X = 0.8  # m/s
#MAX_LIN_X = 0.26  # m/s
MAX_LIN_X = 0.5  # m/s  mini pupper2 推奨速度
#MAX_LIN_Y = 0.4  # m/s
#MAX_LIN_Y = 0.13  # m/s
MAX_LIN_Y = 0.5  # m/s mini pupper2 推奨速度
#MAX_ANG_Z = 2.0  # rad/s
#MAX_ANG_Z = 1.82  # rad/s
MAX_ANG_Z = 1.0  # rad/s mini pupper2 推奨角速度

# joint normalize
# 例：最大可動域を 1.57 rad (90度) と仮定して [-1, 1] に収める場合
MAX_JOINT_RAD = 1.57
MAX_JOINT_RAD20 = 1.57 * 20.0 / 90.0
# 速度もノーマライズ（最大値を15.0 rad/sと仮定）
MAX_JOINT_VEL = 15.0

# 1. 基準となるコントローラーの関節順を定義（クラスの初期化時などに配置）
CONTROLLER_JOINT_ORDER = [
    "base_lf1", "lf1_lf2", "lf2_lf3",  # 左前
    "base_rf1", "rf1_rf2", "rf2_rf3",  # 右前
    "base_lb1", "lb1_lb2", "lb2_lb3",  # 左後
    "base_rb1", "rb1_rb2", "rb2_rb3"   # 右後
]

def euler_from_quaternion_np(q):
    """
    np.float32等のNumPy配列 [x, y, z, w] からオイラー角 (roll, pitch, yaw) へ変換
    """
    x, y, z, w = q
    
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # float32の精度誤差による1超えを防ぐために -1.0 〜 1.0 に丸める
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
        
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

class MiniPupperROSInterface(Node):
    def __init__(self,twist_stamp=False):
        super().__init__("mini_pupper_rl_interface")

        self.twist_stamp=twist_stamp
        self.cmd_vel = np.zeros(3)
        self.cmd_vel_norm = np.zeros(3)

        # ------------------------
        # joint state
        # ------------------------
        self.joint_position = np.zeros(12)
        self.joint_velocity = np.zeros(12)

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10
        )
        #---
        # Ros2 Jazzy 以降は、/cmd_vel は、 TwistStamp に対応する事
        #---
        if not self.twist_stamp:
            self.cmd_pub = self.create_publisher(
                Twist,
                "/cmd_vel",
                10
            )
            self.cmd_sub = self.create_subscription(
                Twist,
                "/cmd_vel",
                self.cmd_vel_callback,
                10
            )
        else:
            self.cmd_pub = self.create_publisher(
                TwistStamped,
                "/cmd_vel",
                10
            )
            self.cmd_sub = self.create_subscription(
                TwistStamped,
                "/cmd_vel",
                self.cmd_vel_callback,
                10
            )
        
        # ------------------------
        # imu
        # ------------------------
        self.quat = np.zeros(4, dtype=np.float32)
        self.quat[3] = 1.0      # add by nishi 2026.7.27
        self.roll = 0.0
        self.pitch = 0.0
        self.pitch_velocity=0.0 # add by nishi 2026.8.10

        # 2. 上限（例：最大5個）を設定して初期化
        # 20msの間に届くROSのメッセージ数（100Hzなら2〜3個）より少し多めにしておけば安全です
        self.pitch_velocity_buffer = deque(maxlen=5) 

        self.imu_sub = self.create_subscription(
            Imu,
            "/imu/data",
            self.imu_callback,
            10
        )
        # ------------------------
        # action publisher
        # ------------------------
        self.command_pub = self.create_publisher(
            Float64MultiArray,
            "simple_quadruped_controller/commands",
            10
        )
        # ------------------------
        # tf
        # ------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )
        # ------------------------
        # gz 空間の roboto pose
        # ------------------------
        # 【追加】軽量化されたポーズトピックを最速で購読する
        self.pose_sub = self.create_subscription(
            PoseArray,
            'mini_pupper_pose',
            self.pose_callback,
            10
        )
        # ------------------------
        # pupper の gz 空間での 位置 add by nishi 2026.7.25
        # ------------------------
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_time = 0.0
        self.last_x = None
        self.last_y = None
        self.last_time = None
        self.current_vx = 0.0
        self.current_vy = 0.0

        self.latest_sim_time=0.0

        self.last_yaw= 0.0

        # ⭕【追加】仮想オドメトリ変数の構築（初期値は原点）
        self.pupper_virt_odom = {
            'x': 0.0,
            'y': 0.0,
            'yaw': 0.0
        }
        # ⭕【追加】命令速度を仮想オドメトリに反映する際の変換係数（factor）
        # シミュレータの挙動に合わせて後から数値を微調整できるように用意
        self.fact_vel = 1.0  # 速度用ファクター
        self.fact_rot = 1.0  # 回転用ファクター

        self.joint_order_fix=True
        self.velocities=None

    # ------------------------
    # callbacks
    # ------------------------
    def cmd_vel_callback(self, msg):
        #print(F"cmd_vel_callback():#1 called!")
        if not self.twist_stamp:
            self.cmd_vel[0] = msg.linear.x
            self.cmd_vel[1] = msg.linear.y
            self.cmd_vel[2] = msg.angular.z
        else:
            self.cmd_vel[0] = msg.twist.linear.x
            self.cmd_vel[1] = msg.twist.linear.y
            self.cmd_vel[2] = msg.twist.angular.z

        self.set_norm_cmd_vel(self.cmd_vel)

    def joint_callback(self,msg):
        #print(F'ros_interface.py::joint_callback()')

        self.velocities = msg.velocity[:12]

        if self.joint_order_fix:
            # 2. 流れてきた名前(name)と現在値(position / velocity)を一対一の辞書にする
            # msg.name, msg.position, msg.velocity のインデックスは常に一致しています
            current_positions = dict(zip(msg.name, msg.position))
            current_velocities = dict(zip(msg.name, msg.velocity))

            # 3. 定義した順番（CONTROLLER_JOINT_ORDER）でデータを抽出し、配列を再構成する
            sorted_positions = []
            sorted_velocities = []

            for joint_name in CONTROLLER_JOINT_ORDER:
                # 辞書から名前をキーにして値を抽出（万が一名前がない場合は0.0を安全値として取得）
                sorted_positions.append(current_positions.get(joint_name, 0.0))
                sorted_velocities.append(current_velocities.get(joint_name, 0.0))
        else:
            sorted_positions = msg.position[:12]
            sorted_velocities = msg.velocity[:12]

        joint_position = np.array(
            #msg.position[:12],
            sorted_positions,
            dtype=np.float32
        )
        # normalize add by nishi 2026.7.27
        clipped_joints = np.clip(joint_position, -MAX_JOINT_RAD, MAX_JOINT_RAD)
        clipped_joints20 = np.clip(joint_position, -MAX_JOINT_RAD20, MAX_JOINT_RAD20)
        self.joint_position = clipped_joints / MAX_JOINT_RAD
        # -20度 から +20度 の部分の補正
        self.joint_position[[0,3,6,9]] = clipped_joints20[[0,3,6,9]] / MAX_JOINT_RAD20
        #self.joint_position[0] = clipped_joints20[0] / MAX_JOINT_RAD20
        #self.joint_position[3] = clipped_joints20[3] / MAX_JOINT_RAD20
        #self.joint_position[6] = clipped_joints20[6] / MAX_JOINT_RAD20
        #self.joint_position[9] = clipped_joints20[9] / MAX_JOINT_RAD20

        joint_velocity = np.array(
            #msg.velocity[:12],
            sorted_velocities,
            dtype=np.float32
        )
        # normalize add by nishi 2026.7.27
        clipped_vel = np.clip(joint_velocity, -MAX_JOINT_VEL, MAX_JOINT_VEL)
        self.joint_velocity = clipped_vel / MAX_JOINT_VEL

    def imu_callback(self,msg):
        self.quat[0] = msg.orientation.x
        self.quat[1] = msg.orientation.y
        self.quat[2] = msg.orientation.z
        self.quat[3] = msg.orientation.w
        self.roll, self.pitch, self.yaw = euler_from_quaternion_np(self.quat)

        self.latest_sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.roll_velocity = msg.angular_velocity.x
        # Y軸の角速度（前後のシーソー運動のスピード）を取得 add by nishi 2026.8.9
        #self.pitch_velocity = np.abs(msg.angular_velocity.y)
        self.pitch_velocity = msg.angular_velocity.y
        self.yaw_velocity = msg.angular_velocity.z

        self.roll_velocity_norm = np.clip(self.roll_velocity * 0.15, -1.0, 1.0)
        self.pitch_velocity_norm = np.clip(self.pitch_velocity * 0.15, -1.0, 1.0)
        self.yaw_velocity_norm = np.clip(self.yaw_velocity * 0.15, -1.0, 1.0)
    
        self.pitch_velocity_buffer.append(self.pitch_velocity)

        # 現在のロール・ヤオの運動エネルギー（角速度の2乗和）をプロパティとして保持
        self.current_motion = self.roll_velocity**2 + self.yaw_velocity**2

        # 2. 【新設・角度】水平からのズレ度合い（絶対値）
        # ロール（左右の傾き）とピッチ（前後の傾き）の絶対値を保持
        self.current_roll_error = np.abs(self.roll)
        self.current_pitch_error = np.abs(self.pitch)

        #print(F'self.pitch_velocity:{self.pitch_velocity:.3f}')

        #q = msg.orientation
        # quaternion -> roll pitch
        #sinr = 2*(q.w*q.x + q.y*q.z)
        #cosr = 1 - 2*(q.x*q.x + q.y*q.y)
        #self.roll = np.arctan2(
        #    sinr,
        #    cosr
        #)
        #sinp = 2*(q.w*q.y - q.z*q.x)
        #self.pitch = np.arcsin(
        #    np.clip(
        #        sinp,
        #        -1.0,
        #        1.0
        #    )
        #)

    def pose_callback(self, msg):
        # poses配列の最初の1個がロボット本体の座標
        if len(msg.poses) > 0:
            # 1. 位置の取得
            self.current_x = msg.poses[0].position.x
            self.current_y = msg.poses[0].position.y
            self.current_z = msg.poses[0].position.z

            # 2. 姿勢（クォータニオン）から、現在の向き（yaw角）をラジアンで計算
            q = msg.poses[0].orientation
            # クォータニオンからZ軸まわりの回転（Yaw）を引っこ抜く簡易数式
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            current_yaw = np.arctan2(siny_cosp, cosy_cosp)

            # シミュレーション時刻
            self.current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # 3. 前進速度(vx) と 旋回速度(vyaw) の超軽量計算
            if self.last_x is not None and self.last_y is not None:
                dt = self.current_time - self.last_time
                if dt > 0.001:
                    # ① まずは「世界（world）基準」の速度を計算する
                    vx_world = (self.current_x - self.last_x) / dt
                    vy_world = (self.current_y - self.last_y) / dt

                    # ② 🔥【重要】世界基準の速度を、現在の向き（current_yaw）を使って「機体（base_link）基準」に変換する
                    cos_yaw = np.cos(current_yaw)
                    sin_yaw = np.sin(current_yaw)

                    # 2次元の回転行列の逆行列（転置）をかけることで、ロボットから見た前・横の速度にする
                    self.current_vx = vx_world * cos_yaw + vy_world * sin_yaw
                    self.current_vy = -vx_world * sin_yaw + vy_world * cos_yaw

                    # 旋回速度（今の向き - 前の向き）
                    # ※-π〜+πの境界をまたぐ時のバグ防止処理
                    dyaw = current_yaw - self.last_yaw
                    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
                    self.current_vyaw = dyaw / dt

                    # ==================================================================
                    # ⭕【新考案】仮想オドメトリ（Pupperのあるべき理想位置）の累積計算
                    # ==================================================================
                    # self.cmd_vel の内訳：[0]: 前進速度(vx), [1]: 横速度(vy), [2]: 旋回速度(v_yaw)
                    cmd_vx = self.cmd_vel[0] * self.fact_vel    # [m/s]
                    cmd_vy = self.cmd_vel[1] * self.fact_vel    # [m/s]
                    cmd_vyaw = self.cmd_vel[2] * self.fact_rot  # [rad/s]
                    
                    # 理想の向き（yaw）の更新
                    self.pupper_virt_odom['yaw'] += cmd_vyaw * dt # [rad/s] * dt -> [rad]
                    # -π 〜 +π の範囲に正規化（バグ防止）
                    self.pupper_virt_odom['yaw'] = np.arctan2(np.sin(self.pupper_virt_odom['yaw']), np.cos(self.pupper_virt_odom['yaw']))
                    
                    # 理想の向きに合わせて、理想の移動距離（X, Y）を世界座標系に累積
                    cos_v = np.cos(self.pupper_virt_odom['yaw'])
                    sin_v = np.sin(self.pupper_virt_odom['yaw'])
                    
                    self.pupper_virt_odom['x'] += (cmd_vx * cos_v - cmd_vy * sin_v) * dt    # [ms] * dt -> [m]
                    self.pupper_virt_odom['y'] += (cmd_vx * sin_v + cmd_vy * cos_v) * dt    # [ms] * dt -> [m]

            self.last_x = self.current_x
            self.last_y = self.current_y
            self.last_z = self.current_z
            self.last_yaw = current_yaw
            self.last_time = self.current_time            

    def set_norm_cmd_vel(self,cmd_vel):
        # 1. 安全のため、まずは設定した最大値でクリップする（異常値対策）
        clipped_linear_x = np.clip(cmd_vel[0], -MAX_LIN_X, MAX_LIN_X)
        clipped_linear_y = np.clip(cmd_vel[1], -MAX_LIN_Y, MAX_LIN_Y)
        clipped_angular_z = np.clip(cmd_vel[2], -MAX_ANG_Z, MAX_ANG_Z)

        # 2. 最大値で割って [-1, 1] の範囲に正規化する
        self.cmd_vel_norm = np.array([
            clipped_linear_x / MAX_LIN_X,
            clipped_linear_y / MAX_LIN_Y,
            clipped_angular_z / MAX_ANG_Z
        ])

    # ------------------------
    # observation
    # ------------------------
    def get_observation(self):
        obs = np.concatenate(
            [
                self.cmd_vel_norm,     # 3
                self.joint_position,   # 12
                self.joint_velocity,   # 12  ←追加！
                #[
                #    self.roll,      # 1
                #    self.pitch      # 1
                #]
                self.quat,       # 4
                [
                    self.roll_velocity_norm,
                    self.pitch_velocity_norm,  # 💡 Y を先に配置
                    self.yaw_velocity_norm,    # 💡 Z を最後に配置
                ],
            ]
        )
        return obs.astype(
            np.float32
        )
    # ------------------------
    # action
    # ------------------------
    def send_action(self,action):
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.command_pub.publish(
            msg
        )

    # Gymのstepから呼ばれる関数を、変数横流しだけの超軽量処理にする
    def get_forward_velocity(self):
        #return getattr(self, 'current_vx', 0.0)
        return self.current_vx

    def get_side_velocity(self):
        #return getattr(self, 'current_vy', 0.0)
        return self.current_vy

    def get_yaw_velocity(self):
        #return getattr(self, 'current_vyaw', 0.0)
        return self.current_vyaw

    def get_pitch_velocity(self):
        if len(self.pitch_velocity_buffer) > 0:
            abs_pitch_vel = np.mean(self.pitch_velocity_buffer)
            # 次の20msのためにバッファを空にする
            self.pitch_velocity_buffer.clear()
        else:
            abs_pitch_vel = 0.0
        return abs_pitch_vel

    #def wait(self):
    #    rclpy.spin_once(
    #        self,
    #        timeout_sec=0.01
    #    )

    def wait_for_gazebo_steps(self, target_steps=1, call_th=False):
        """ 指定したステップ数分、Gazeboの時間（シミュレーション時間）が進むまで待つ """
        start_sim_time = self.latest_sim_time
        dur_time = target_steps * 10.0 / 1000.0  # target_steps=1 なら 0.01秒
        # 1000回のループ上限でフリーズを確実に防ぐ（タイムアウトセーフティ）
        for _ in range(1000):
            if not call_th:
                rclpy.spin_once(self, timeout_sec=0.001)
            else:
                # 💡 spin_once は呼ばず、メインスレッドがデータを更新してくれるのを 1ms ずつ待つ
                import time
                time.sleep(0.001)

            # Gazebo内の経過時間が目標値に達したら即座にループを抜ける
            if (self.latest_sim_time - start_sim_time) >= dur_time:
                break

    def publish_cmd_vel_old(self, vx, vy, wz):
        if not self.twist_stamp:
            msg = Twist()
            msg.linear.x = float(vx)
            msg.linear.y = float(vy)
            msg.angular.z = float(wz)
        else:
            msg = TwistStamped()
            msg.twist.linear.x = float(vx)
            msg.twist.linear.y = float(vy)
            msg.twist.angular.z = float(wz)
            # 1. ヘッダー情報の設定
            msg.header.stamp = self.get_clock().now().to_msg() # 現在時刻のタイムスタンプ
            msg.header.frame_id = 'base_link'                  # ロボットの基準座標系

        self.cmd_pub.publish(msg)

    def publish_cmd_vel(self, cv):
        if not self.twist_stamp:
            msg = Twist()
            # cv[vx, vy, v_yaw]
            msg.linear.x = float(cv[0])
            msg.linear.y = float(cv[1])
            msg.angular.z = float(cv[2])
        else:
            msg = TwistStamped()
            # cv[vx, vy, v_yaw]
            msg.twist.linear.x = float(cv[0])
            msg.twist.linear.y = float(cv[1])
            msg.twist.angular.z = float(cv[2])

            # 1. ヘッダー情報の設定
            msg.header.stamp = self.get_clock().now().to_msg() # 現在時刻のタイムスタンプ
            msg.header.frame_id = 'base_link'                  # ロボットの基準座標系

        self.cmd_pub.publish(msg)

    # --------------------------------------------------
    # 【追加】Gazeboのシミュレーション時間を「秒（float）」で返す関数
    # --------------------------------------------------
    def get_sim_time(self):
        # ROS 2ノードが持っている現在のクロック（シミュレーション時間）を取得
        now = self.get_clock().now()
        # ナノ秒を秒（少数点付きのfloat）に変換して返す（例: 54.304）
        return now.nanoseconds / 1e9

    # --------------------------------------------------
    # 【追加】ワープリセット時に、古い記憶を強制的に消去する関数
    # --------------------------------------------------
    def reset_internal_states(self):
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vyaw = 0.0
        self.last_yaw = 0.0  # ⭕ ここが最重要！古いヨコ向きの記憶を消去
        self.last_x = None   # 速度計算の基準も一旦クリア
        self.last_y = None   # 速度計算の基準も一旦クリア
        self.last_time = None

        # ⭕【追加】ワープ時に仮想オドメトリも完全に原点へリセット
        self.pupper_virt_odom['x'] = 0.0
        self.pupper_virt_odom['y'] = 0.0
        self.pupper_virt_odom['yaw'] = 0.0

        self.roll_velocity_norm=0.0
        self.pitch_velocity_norm=0.0  # 💡 Y を先に配置
        self.yaw_velocity_norm=0.0    # 💡 Z を最後に配置

        self.velocities=None
