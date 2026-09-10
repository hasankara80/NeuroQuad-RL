import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces


class DomainRandomizedQuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path="assets/quadruped.xml", render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        obs_dim = 1 + 4 + 8 + 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.base_torso_mass = float(self.model.body_mass[1])
        self.base_friction = float(self.model.geom_friction[0][0])

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _randomize_domain(self):
        mass_variance = np.random.uniform(0.8, 1.2)
        self.model.body_mass[1] = self.base_torso_mass * mass_variance

        friction_variance = np.random.uniform(0.7, 1.3)
        self.model.geom_friction[0][0] = self.base_friction * friction_variance

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        obs = np.concatenate([
            [qpos[2]],
            qpos[3:7],
            qpos[7:],
            qvel[6:]
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 🛑 NEW: Set a "nominal posture" (Hips slightly forward, knees bent backward)
        # This prevents the robot from spawning stiff-legged and instantly collapsing.
        nominal_posture = np.array([
            0.2, -0.8,  # Front Left (Hip, Knee)
            0.2, -0.8,  # Front Right (Hip, Knee)
            0.2, -0.8,  # Back Left (Hip, Knee)
            0.2, -0.8  # Back Right (Hip, Knee)
        ])
        self.data.qpos[7:] = nominal_posture

        # Apply Domain Randomization
        self._randomize_domain()

        # Initial forward step
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human" and hasattr(self, 'viewer'):
            self.viewer.sync()

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # SAFETY CATCH: If the AI breaks the physics engine, end the episode immediately
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            # Give a massive penalty for breaking physics
            return np.zeros(self.observation_space.shape, dtype=np.float32), -50.0, True, False, {}

        obs = self._get_obs()

        forward_vel = self.data.qvel[0]
        torso_z = self.data.qpos[2]
        qw = obs[1]  # The 'w' component of the orientation quaternion

        # 1. ALIVE BONUS: Reward the robot just for surviving the step upright
        alive_bonus = 1.0

        # 2. ACTION PENALTY: Reduced from 0.1 to 0.05 so it isn't afraid to use torque
        action_penalty = 0.05 * np.sum(np.square(action))

        # 3. ORIENTATION PENALTY: Reduced slightly so it can lean into its walk
        orientation_penalty = 1.0 * (1.0 - qw)

        # 4. FORWARD REWARD: Increased multiplier to make walking highly desirable
        reward = (forward_vel * 2.5) + alive_bonus - action_penalty - orientation_penalty
        done = False

        # Fall & Flip detection
        if torso_z < 0.15 or qw < 0.8:
            reward -= 10.0
            done = True

        if self.render_mode == "human" and hasattr(self, 'viewer'):
            self.viewer.sync()

        return obs, reward, done, False, {}

    def close(self):
        if hasattr(self, 'viewer'):
            self.viewer.close()