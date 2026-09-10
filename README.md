# NeuroQuad-RL: PPO Quadruped Locomotion in MuJoCo

**NeuroQuad-RL** is an end-to-end reinforcement-learning prototype for continuous quadruped control in MuJoCo.

The project combines a custom 8-DOF quadruped robot, a Gymnasium environment, PPO training with Stable-Baselines3, per-episode domain randomisation, deterministic simulation evaluation, ONNX actor-policy export, and ROS 2-oriented inference tooling.

> **Current scope:** Training and evaluation take place in MuJoCo simulation. ONNX export, Docker packaging, and a ROS 2 inference interface are included, but deployment to physical quadruped hardware and sim-to-real transfer have not yet been validated.

---

## 🧠 Pipeline Architecture

```text
Custom MuJoCo Quadruped
        ↓
Gymnasium Environment
        ↓
Domain-Randomised PPO Training
        ↓
Deterministic Simulation Evaluation
        ↓
ONNX Actor-Policy Export
        ↓
ROS 2-Oriented Inference Interface
```

### Implemented Components

- **Custom Simulation:** Defines an 8-DOF quadruped using a MuJoCo XML model.
- **Gymnasium Environment:** Provides continuous observation and action spaces compatible with reinforcement-learning libraries.
- **Domain Randomisation:** Randomises torso mass and floor friction at the beginning of each episode.
- **PPO Training:** Trains an MLP policy using Stable-Baselines3.
- **Checkpointing:** Saves intermediate PPO checkpoints every 500,000 timesteps.
- **Simulation Evaluation:** Runs deterministic policy actions in the MuJoCo viewer.
- **ONNX Export:** Extracts and exports the actor-policy pathway for portable inference.
- **Deployment Scaffolding:** Packages ONNX Runtime inference and a Python ROS 2 interface using Docker.

---

## 🚀 Key Features

- Custom 8-DOF quadruped model in MuJoCo
- Continuous-control Gymnasium environment
- PPO training with Stable-Baselines3
- Per-episode mass and friction randomisation
- Forward-motion reward and uprightness constraints
- Fall, excessive-tilt, and invalid-physics termination
- Nominal initial joint posture
- Deterministic simulation evaluation
- Periodic model checkpointing
- TensorBoard-compatible training logs
- PyTorch actor-policy export to ONNX
- Dynamic batch support in the exported ONNX model
- Docker packaging with ROS 2 Humble
- Python-based ONNX Runtime inference tooling

---

## 🎛️ Observation and Action Spaces

### Observation Space

The policy receives a 21-dimensional observation vector:

| Component | Dimensions |
|---|---:|
| Torso height | 1 |
| Torso orientation quaternion | 4 |
| Actuated joint positions | 8 |
| Actuated joint velocities | 8 |
| **Total** | **21** |

The observation is created from the relevant MuJoCo position and velocity values and converted to `float32`.

### Action Space

The policy produces eight continuous actuator commands:

```text
Box(low=-1.0, high=1.0, shape=(8,), dtype=float32)
```

The commands are sent through MuJoCo's actuator-control interface:

```python
data.ctrl[:] = action
```

The physical meaning of each control value depends on the actuator definitions in:

```text
assets/quadruped.xml
```

---

## 🎯 Reward Design

The reward function combines forward locomotion, survival, control efficiency, and orientation.

```text
reward =
    2.5 × forward velocity
    + 1.0 alive bonus
    - 0.05 × squared action magnitude
    - orientation penalty
```

### Reward Components

- **Forward-velocity reward:** Encourages movement in the positive forward direction.
- **Alive bonus:** Rewards the quadruped for remaining active and upright.
- **Action penalty:** Discourages unnecessarily large actuator commands.
- **Orientation penalty:** Penalises deviation from the desired torso orientation.

### Episode Termination

An additional penalty of `-10.0` is applied and the episode terminates when:

- The torso height falls below `0.15 m`, or
- The configured orientation check falls below its safety threshold.

A penalty of `-50.0` is applied if MuJoCo produces NaN values in the joint positions or velocities.

---

## 🐛 Reward-Hacking Case Study

During early development, the PPO policy discovered an unintended strategy. Instead of learning stable forward locomotion, the quadruped could rotate onto its back and move its legs while avoiding part of the original fall condition.

This behaviour demonstrated an important reinforcement-learning problem:

> Optimising the written reward is not always equivalent to learning the intended behaviour.

### Mitigation

The environment was revised to include:

- Forward-velocity reward
- Upright survival bonus
- Action-magnitude penalty
- Orientation penalty
- Torso-height termination
- Excessive-tilt termination
- Invalid-physics detection
- A nominal bent-leg starting posture

The nominal posture helps the robot absorb the initial drop forces rather than beginning with fully extended legs.

### Optional Failure Demonstration

A GIF showing the early reward-hacking behaviour can be added here:

```text
assets/reward_hacking_failure.gif
```

Suggested caption:

> **Reward-hacking failure mode:** An early policy exploited the termination condition by rotating onto its back instead of learning stable forward locomotion.

---

## 🎲 Domain Randomisation

NeuroQuad-RL randomises selected physical parameters at each environment reset.

| Parameter | Randomisation range |
|---|---:|
| Torso mass multiplier | `0.8×` to `1.2×` |
| Floor-friction multiplier | `0.7×` to `1.3×` |

The purpose of domain randomisation is to expose the policy to variations in simulated dynamics and reduce dependence on one fixed environment configuration.

Domain randomisation can support future transfer experiments, but its inclusion alone does not demonstrate successful sim-to-real transfer.

---

## 🤖 PPO Training Configuration

The current training configuration uses:

| Parameter | Value |
|---|---:|
| Algorithm | PPO |
| Policy | MLP Policy |
| Requested training timesteps | 10,000,000 |
| Learning rate | `3e-4` |
| Rollout steps | 2,048 |
| Batch size | 128 |
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |
| Checkpoint interval | 500,000 timesteps |
| Vector environment | One `DummyVecEnv` environment |

The training script requests 10 million timesteps. This is a configured target, not a measured result unless the complete training run finishes successfully.

---

## 📊 Current Implementation Status

| Component | Status |
|---|---|
| Custom MuJoCo quadruped | Implemented |
| Gymnasium environment | Implemented |
| PPO training pipeline | Implemented |
| Mass randomisation | Implemented |
| Friction randomisation | Implemented |
| Periodic checkpointing | Implemented |
| Deterministic simulation evaluation | Implemented |
| ONNX actor-policy export | Implemented |
| Docker packaging | Implemented |
| ROS 2 inference scaffolding | Implemented |
| Multi-seed benchmarking | Not yet reported |
| Aggregate locomotion metrics | Not yet reported |
| ONNX inference benchmark | Not yet reported |
| Physical quadruped deployment | Not yet evaluated |
| Sim-to-real validation | Not yet evaluated |

Quantitative locomotion results are not currently reported. Future evaluation should include mean episode reward, forward velocity, forward distance, episode duration, fall rate, and results across multiple random seeds.

---

## 📦 Installation

### Prerequisites

- Python 3.10 or 3.11
- MuJoCo-compatible operating system
- Conda, Miniforge, or another Python environment manager
- Docker, required only for containerised ROS 2 inference
- A graphical desktop environment for interactive MuJoCo evaluation

### 1. Clone the Repository

```bash
git clone https://github.com/hasankara80/NeuroQuad-RL.git
cd NeuroQuad-RL
```

### 2. Create a Python Environment

Using Conda:

```bash
conda create -n neuroquad-rl python=3.11 -y
conda activate neuroquad-rl
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A suitable `requirements.txt` should include tested versions of:

```text
gymnasium
mujoco
numpy
stable-baselines3
torch
onnx
onnxruntime
tensorboard
```

---

## 🚀 Execution Guide

### 1. Train the PPO Policy

```bash
python train_ppo.py
```

The training script:

1. Loads the custom MuJoCo environment.
2. Validates the environment with Stable-Baselines3.
3. Creates a PPO MLP policy.
4. Trains for the configured number of timesteps.
5. Saves periodic checkpoints.
6. Writes TensorBoard-compatible logs.
7. Saves the final policy.

The final Stable-Baselines3 model is saved to:

```text
models/ppo_quadruped_final.zip
```

Intermediate checkpoints are saved to:

```text
models/checkpoints/
```

### 2. Monitor Training

If TensorBoard is installed, run:

```bash
tensorboard --logdir rl_logs
```

Then open the local TensorBoard URL shown in the terminal.

### 3. Evaluate the Policy in MuJoCo

```bash
python evaluate_policy.py
```

The evaluation script:

- Loads the trained PPO checkpoint
- Creates the domain-randomised environment
- Executes deterministic policy actions
- Displays the quadruped in the MuJoCo viewer
- Resets the environment when an episode terminates

Press `Control + C` in the terminal to stop the evaluation.

### 4. Export the Actor Policy to ONNX

```bash
python export_onnx.py
```

The exported model is saved to:

```text
models/quadruped_policy.onnx
```

### 5. Build the ROS 2 Inference Container

```bash
docker build -t neuroquad-rl .
```

### 6. Run the Container

```bash
docker run --rm neuroquad-rl
```

The Docker image provides ROS 2 Humble and Python-based ONNX Runtime inference scaffolding. Physical motor drivers and a hardware-specific control interface are not included.

---

## 📦 ONNX Export

The export script isolates the actor-policy pathway from the complete Stable-Baselines3 PPO model.

### Exported Interface

- **Input name:** `observation`
- **Input shape:** `(batch_size, 21)`
- **Output name:** `action`
- **Output shape:** `(batch_size, 8)`
- **ONNX opset:** 17
- **Output path:** `models/quadruped_policy.onnx`

Dynamic axes allow the exported graph to accept different batch sizes.

The ONNX export is intended to reduce dependency on the original Stable-Baselines3 training workflow during inference.

Before hardware use, the ONNX outputs should be quantitatively compared with Stable-Baselines3 outputs using identical observations.

---

## 🐳 Docker and ROS 2 Integration

The Dockerfile uses:

```text
osrf/ros:humble-desktop
```

The image installs:

- Python package management
- ONNX Runtime
- NumPy
- The exported ONNX policy
- A Python ROS 2 inference node

The container packages the policy-inference layer but does not currently include:

- Physical quadruped drivers
- Hardware-specific actuator messages
- Sensor integration
- A real-time operating system
- Safety monitoring
- Emergency-stop handling
- Physical deployment validation

The ROS 2 component should therefore be understood as deployment scaffolding rather than a completed hardware-control system.

---

## 📁 Repository Structure

```text
NeuroQuad-RL/
├── assets/
│   └── quadruped.xml              # MuJoCo quadruped model
├── deployment/
│   └── ros2_inference_node.py     # Python ROS 2 inference interface
├── models/
│   ├── checkpoints/               # Intermediate PPO checkpoints
│   ├── ppo_quadruped_final.zip    # Final Stable-Baselines3 model
│   └── quadruped_policy.onnx      # Exported actor policy
├── rl_logs/                       # TensorBoard training logs
├── quadruped_env.py               # Gymnasium MuJoCo environment
├── train_ppo.py                   # PPO training pipeline
├── evaluate_policy.py             # Deterministic simulation evaluation
├── export_onnx.py                 # PyTorch-to-ONNX actor export
├── Dockerfile                     # ROS 2 and ONNX Runtime container
├── requirements.txt               # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

Generated logs, caches, checkpoints, and local IDE files should normally be excluded from source control.

---

## ⚠️ Limitations

NeuroQuad-RL is currently a simulation-based research and portfolio prototype.

- Training and evaluation currently run in MuJoCo.
- Physical quadruped deployment has not been demonstrated.
- Sim-to-real transfer has not been validated.
- Domain randomisation is currently limited to torso mass and floor friction.
- The training script uses one `DummyVecEnv` environment rather than massively parallel simulation.
- The evaluation script does not currently report aggregate locomotion metrics.
- Training results have not yet been reported across multiple random seeds.
- The current orientation penalty uses one quaternion component and may not provide a complete uprightness measure for every orientation.
- The exported ONNX policy has not yet been quantitatively verified against Stable-Baselines3 inference.
- ONNX inference latency has not yet been benchmarked.
- The ROS 2 code is deployment scaffolding and is not connected to documented physical motor hardware.
- Hardware safety, actuator limits, communication latency, and emergency-stop behaviour have not been evaluated.

---

## 🗺️ Roadmap

### Completed

- [x] Custom MuJoCo quadruped model
- [x] Gymnasium environment
- [x] Twenty-one-dimensional observation space
- [x] Eight-dimensional continuous action space
- [x] PPO training pipeline
- [x] Torso-mass randomisation
- [x] Floor-friction randomisation
- [x] Reward-shaping iteration
- [x] Deterministic simulation evaluation
- [x] Stable-Baselines3 checkpointing
- [x] ONNX actor-policy export
- [x] Docker packaging
- [x] ROS 2 inference scaffolding

### Planned Improvements

- [ ] Add multi-seed training and evaluation
- [ ] Report mean episode reward
- [ ] Report mean episode duration
- [ ] Report average forward velocity and distance
- [ ] Report episode fall rate
- [ ] Add vectorised parallel training environments
- [ ] Replace the quaternion-component orientation check with a torso up-vector metric
- [ ] Expand dynamics randomisation
- [ ] Add observation noise and latency simulation
- [ ] Compare Stable-Baselines3 and ONNX outputs
- [ ] Benchmark ONNX inference latency
- [ ] Add automated environment and export tests
- [ ] Document ROS 2 message interfaces
- [ ] Integrate a hardware-specific actuator interface
- [ ] Evaluate deployment on physical quadruped hardware
- [ ] Conduct formal sim-to-real validation

---

## 🧪 Recommended Future Evaluation

A stronger experimental evaluation should report:

- At least three independent random seeds
- Training return across timesteps
- Mean and standard deviation of episode return
- Mean forward velocity
- Mean forward distance
- Mean episode duration
- Fall rate
- Performance under nominal simulation parameters
- Performance across randomised mass and friction settings
- Stable-Baselines3 versus ONNX action difference
- Average ONNX inference latency
- Worst-case ONNX inference latency

Physical-robot results should be reported separately from simulation results.

---

## 🙏 Acknowledgements

This project uses:

- MuJoCo for physics simulation
- Gymnasium for the reinforcement-learning environment interface
- Stable-Baselines3 for PPO training
- PyTorch for neural-network execution and export
- ONNX and ONNX Runtime for portable policy inference
- ROS 2 Humble for deployment-oriented middleware tooling

---

## 📬 Feedback

Feedback is welcome on:

- Quadruped reward design
- Domain-randomisation strategy
- Locomotion evaluation
- PPO training configuration
- ONNX actor export
- ROS 2 inference integration
- Future sim-to-real evaluation
