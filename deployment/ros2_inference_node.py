# noinspection PyUnresolvedReferences
import rclpy
# noinspection PyUnresolvedReferences
from rclpy.node import Node
# noinspection PyUnresolvedReferences
from sensor_msgs.msg import JointState, Imu
# noinspection PyUnresolvedReferences
from std_msgs.msg import Float32MultiArray
import onnxruntime as ort
import numpy as np


class QuadrupedRLController(Node):
    """
    ROS2 Node for Real-Time RL Inference.
    Subscribes to robot sensor telemetry, runs the ONNX optimized PPO policy,
    and publishes motor targets at 100Hz.
    """

    def __init__(self):
        super().__init__('quadruped_rl_controller')

        self.get_logger().info("🚀 Initializing RL Quadruped Controller...")

        # 1. Load the ONNX Inference Session (CPU optimized for robotics hardware)
        self.ort_session = ort.InferenceSession(
            "models/quadruped_policy.onnx",
            providers=['CPUExecutionProvider']
        )
        self.get_logger().info("✅ ONNX Policy Graph Loaded.")

        # State buffers
        self.joint_pos = np.zeros(8)
        self.joint_vel = np.zeros(8)
        self.torso_z = np.zeros(1)
        self.orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

        # 2. Subscribers (Listening to hardware sensors)
        self.joint_sub = self.create_subscription(
            JointState, '/quadruped/joint_states', self.joint_cb, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/quadruped/imu', self.imu_cb, 10)

        # 3. Publisher (Sending commands to motors)
        self.motor_pub = self.create_publisher(
            Float32MultiArray, '/quadruped/motor_cmds', 10)

        # 4. High-Speed Control Loop (100Hz)
        self.timer = self.create_timer(0.01, self.control_loop)

    def joint_cb(self, msg):
        """Callback to update joint positions and velocities from hardware."""
        self.joint_pos = np.array(msg.position[:8])
        self.joint_vel = np.array(msg.velocity[:8])

    def imu_cb(self, msg):
        """Callback to update orientation and height from hardware IMU/Odometry."""
        self.orientation_quat = np.array([
            msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z
        ])
        # Note: In a real system, Z-height comes from odometry or state estimation
        self.torso_z = np.array([0.3])

    def control_loop(self):
        """Runs the ONNX graph and publishes motor torques."""
        # Construct the 21-dimensional observation vector expected by the policy
        obs = np.concatenate([
            self.torso_z,
            self.orientation_quat,
            self.joint_pos,
            self.joint_vel
        ]).astype(np.float32)

        # Add batch dimension (1, 21)
        obs_input = np.expand_dims(obs, axis=0)

        # Run ultra-fast ONNX inference
        ort_inputs = {self.ort_session.get_inputs()[0].name: obs_input}
        action = self.ort_session.run(None, ort_inputs)[0]

        # Publish the 8 motor commands to the hardware
        msg = Float32MultiArray()
        msg.data = action[0].tolist()
        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrupedRLController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RL Controller.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()