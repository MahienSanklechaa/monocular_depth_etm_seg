# Real-Time Joint Depth Estimation & Semantic Segmentation

An ultra-lightweight, multi-task convolutional neural network designed for real-time edge inference. By pairing a MobileNetV2 backbone with a custom PixelShuffle decoder and dense skip connections, this architecture achieves high-fidelity monocular depth estimation and semantic segmentation simultaneously under strict resource constraints.

# 🚀 Performance Highlights (KITTI Benchmarks)
* **Model Size Reduction:** **18.4x** smaller than standard real-time baseline models.
* **Parameter Count:** Compressed down to just **0.32M parameters**.
* **Depth Accuracy:** Achieved **3.871m RMSE** with a **99.8% depth threshold accuracy** ($\delta < 1.56$).

# 🏗️ Architecture Overview

The network follows an asymmetrical Encoder-Decoder topology optimized for joint multi-task prediction:
1. **Encoder:** MobileNetV2 backbone extracting multi-scale feature representations. Feature map channels are captured via four dense skip connections to preserve fine spatial details.
2. **Decoder:** A custom sub-pixel convolution (`nn.PixelShuffle`) upscaling pipeline that eliminates the heavy computational footprint of standard transpose convolutions.
3. **Task Heads:** Separate, detached convolutional layers outputting 1-channel continuous metric depth values and a 19-channel semantic segmentation map.

# 📉 Core Methodologies & Loss Function

To stabilize the joint optimization of disparate depth landscapes and semantic labels, the model utilizes a specialized dual-objective objective function:

$$L_{total} = \alpha \cdot L_{depth} + \beta \cdot L_{seg}$$

* **BerHu-Normalized Log-Space Depth Loss ($L_{depth}$):** Depth ground truths and predictions are mapped into a normalized log-space to compress the target variance. A Reverse Huber (BerHu) penalty is applied to aggressively weight near-field estimation errors, which are critical for autonomous navigation and obstacle avoidance.
* **Task Balancing:** Managed using an asymmetrical weight allocation ($\alpha = 0.25$, $\beta = 0.75$) to prevent semantic gradient dominance over the regression task.

# ⚙️ Repository Structure & Setup

# Prerequisites
Ensure you have the following packages installed:
```bash
pip install torch torchvision numpy scipy matplotlib pillow