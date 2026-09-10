# NeuroQuad-RL: End-to-End Sim-to-Real Quadruped Locomotion

An end-to-end Reinforcement Learning pipeline for quadruped locomotion. This project demonstrates a production-grade RL workflow: from massively parallel simulation and domain randomization to ONNX compilation and ROS2 hardware deployment.

*(Note: On GitHub, edit this file and drag-and-drop your successful landing video right here!)*

## 🧠 Pipeline Architecture

`MuJoCo (Physics)` ➔ `Gymnasium (Domain Randomization)` ➔ `PPO (Training)` ➔ `ONNX (Compilation)` ➔ `ROS2 (Deployment)`

*   **Custom Simulation:** Built an 8-DOF quadruped robot in XML. Transitioned from raw torque to PD Position Control to emulate real-world actuator hardware and utilized a nominal posture to absorb initial drop forces.
*   **Sim-to-Real Transfer:** Engineered dynamic Domain Randomization within the Gymnasium environment, independently modulating payload mass and floor friction per episode to prevent the policy from overfitting to the simulator.
*   **Production Deployment:** Decoupled the trained PyTorch actor network from Python by compiling it into a microsecond-latency C++ ONNX graph. Containerized the inference graph inside a real-time ROS2 `rclpy` node via Docker.

## 🐛 Overcoming "Reward Hacking"

During initial training, the PPO agent discovered a local optimum: to avoid the -10.0 penalty for the torso dropping below `0.15m`, the robot would immediately flip onto its back (the thickest part of its body) and wiggle its legs in the air, ensuring it technically never "fell."

*(Note: On GitHub, edit this file and drag-and-drop your turtle-flip blooper video right here!)*

**The Fix:** I implemented strict Reward Shaping by introducing an orientation penalty based on the quaternion's `w` component, and immediately terminating the episode if the robot tilted past a safe threshold. 

## 🚀 Execution Guide

**1. Train the Policy**
```bash
python train_ppo.py