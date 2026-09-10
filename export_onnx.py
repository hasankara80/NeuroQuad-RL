import torch
import torch.nn as nn
from stable_baselines3 import PPO


class OnnxablePolicy(nn.Module):
    """
    A PyTorch wrapper that extracts ONLY the deterministic actor network.
    This strips away all the RL training bloat (the Critic network,
    value functions, and random distributions) leaving just a pure,
    lightning-fast inference graph.
    """

    def __init__(self, extractor, action_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net

    def forward(self, observation):
        # 1. Pass observation through the Multi-Layer Perceptron (MLP)
        action_hidden, _ = self.extractor(observation)
        # 2. Map hidden layers to the 8 motor torques
        return self.action_net(action_hidden)


def main():
    print("🔄 Loading PyTorch RL Policy...")
    # Load the model on CPU for export
    model = PPO.load("models/ppo_quadruped_final", device="cpu")

    # Wrap the internal network components
    onnxable_model = OnnxablePolicy(
        model.policy.mlp_extractor,
        model.policy.action_net
    )

    # Create a dummy observation to trace the graph
    # Our observation space has 21 dimensions (1 Z-height + 4 Quat + 8 QPos + 8 QVel)
    dummy_input = torch.randn(1, 21)

    onnx_path = "models/quadruped_policy.onnx"
    print(f"⚙️ Compiling PyTorch model to ONNX graph...")

    # Export the graph
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"}
        }
    )

    print(f"✅ ONNX Export Complete! Highly-optimized inference graph saved to: {onnx_path}")


if __name__ == "__main__":
    main()