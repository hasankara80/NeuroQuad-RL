import time
from stable_baselines3 import PPO
from quadruped_env import DomainRandomizedQuadrupedEnv


def main():
    print("🤖 Loading Trained PPO Policy...")

    # 1. Load the environment with the 3D viewer enabled
    env = DomainRandomizedQuadrupedEnv(render_mode="human")

    # 2. Load the trained PyTorch weights
    model = PPO.load("models/ppo_quadruped_final", env=env)

    obs, _ = env.reset()

    print("📺 Simulating Policy (Press CTRL+C in terminal to stop)...")
    for _ in range(2000):
        # The AI evaluates the observation and decides on the optimal motor torques
        # deterministic=True forces it to use its best learned policy, removing randomness
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)

        # Slow down the loop slightly so our human eyes can actually watch it
        time.sleep(0.01)

        if done:
            obs, _ = env.reset()

    env.close()
    print("✅ Evaluation Complete.")


if __name__ == "__main__":
    main()