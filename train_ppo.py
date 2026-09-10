import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from quadruped_env import DomainRandomizedQuadrupedEnv


def main():
    print("🔍 Validating Custom MuJoCo Environment...")
    raw_env = DomainRandomizedQuadrupedEnv(render_mode=None)
    check_env(raw_env, warn=True)

    vec_env = DummyVecEnv([lambda: DomainRandomizedQuadrupedEnv(render_mode=None)])

    print("🤖 Initializing PPO Neural Network...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,  # Increased from 64 for more stable gradient updates
        gamma=0.99,  # Discount factor (standard for MuJoCo)
        gae_lambda=0.95,
        ent_coef=0.01,  # 🛑 NEW: Entropy coefficient forces the AI to explore
        verbose=1,
        tensorboard_log="./rl_logs/"
    )

    # 🛑 NEW: Save a checkpoint every 500,000 steps so you don't lose progress!
    os.makedirs("models/checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path='./models/checkpoints/',
        name_prefix='ppo_quadruped'
    )

    # 🚀 NEW: Train for 10 Million Steps (~1 hour on Apple Silicon)
    training_steps = 10_000_000
    print(f"🚀 Starting Training for {training_steps} timesteps...")

    model.learn(
        total_timesteps=training_steps,
        callback=checkpoint_callback,
        tb_log_name="PPO_Quadruped_Run2"
    )

    # Save final model
    save_path = "models/ppo_quadruped_final"
    model.save(save_path)
    print(f"🎉 Training Complete! Policy saved to: {save_path}.zip")


if __name__ == "__main__":
    main()