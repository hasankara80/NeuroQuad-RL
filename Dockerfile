# Base image: Official ROS2 Humble
FROM osrf/ros:humble-desktop

# Set working directory
WORKDIR /app

# Install dependencies for inference
RUN apt-get update && apt-get install -y \
    python3-pip \`
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime and Numpy
RUN pip3 install onnxruntime numpy

# Copy the ONNX model and ROS2 Node into the container
COPY models/quadruped_policy.onnx /app/models/
COPY deployment/ros2_inference_node.py /app/deployment/

# Source ROS2 and run the node
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && python3 deployment/ros2_inference_node.py"]