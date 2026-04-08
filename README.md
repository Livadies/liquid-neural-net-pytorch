# 🌊 Liquid Neural Networks (LNN) in PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Status](https://img.shields.io/badge/Status-Research_&_Development-success)

A custom, from-scratch implementation of **Liquid Neural Networks (LNN)** in PyTorch. This repository demonstrates how continuous-time neural networks governed by Ordinary Differential Equations (ODEs) can outperform discrete RNNs/LSTMs in highly volatile environments, such as algorithmic trading and financial time series.

## 🧠 Why Liquid Networks?

Traditional Recurrent Neural Networks (RNNs, LSTMs, GRUs) operate in discrete time. They treat the temporal gap between steps (e.g., 1 millisecond vs. 3 days) as identical indices. In environments with extreme noise and sudden volatility spikes (like crypto or HFT markets), discrete models often fail to capture the true physical flow of time, either overfitting to noise or converging to a trivial moving average.

LNNs solve this by modeling the hidden state $h(t)$ as a continuous variable using an ODE:

$$\frac{dh(t)}{dt} = -\frac{h(t)}{\tau} + f(W_{in} x(t) + W_h h(t) + b)$$

Where $\tau$ is a learnable time-constant (viscosity/persistence) parameter for each neuron. 
* **Low $\tau$ (Fast Neurons):** React instantly to high-frequency market noise.
* **High $\tau$ (Slow Neurons):** Ignore the noise and maintain long-term macro-trend context.

## 🛠️ Implementation Details

The core of this repo is the `LiquidLayer`. It solves the differential equation using the Euler integration method (`dt`) directly within the PyTorch `forward` pass.

**Key Features:**
- **Custom Learnable $\tau$:** The network autonomously learns which neurons should be fast and which should be slow.
- **Exploding Gradient Protection:** Implemented `torch.clamp` on the $\tau$ parameter to prevent divisions by zero during backpropagation.
- **Multi-GPU Support:** Fully compatible with `torch.nn.DataParallel` for massive batch processing.

## 📊 Experiment: Decoding Market Chaos

To prove the LNN's effectiveness, we created a synthetic financial dataset simulating Geometric Brownian Motion with severe volatility jumps and a hidden macro-trend signal.

While standard binary classification approaches get stuck at ~50% accuracy (Loss: 0.693), our `LiquidNet` successfully filters the high-frequency chaos.

* **Test Accuracy:** Achieved **83.3%** in predicting the next market phase.
* **Neuron Viscosity Distribution:** The network naturally developed "fast" neurons ($\tau \approx 0.54$) and "slow" context neurons ($\tau \approx 2.54$).

### Visualizing the Liquid State

*(Insert your Kaggle charts here)*
> **Top:** The noisy input signal (Market Chaos) vs LNN Prediction.
> **Bottom:** The continuous hidden states $h(t)$. Notice how the orange/red curves (high $\tau$) build a smooth trend, while blue/purple curves (low $\tau$) vibrate with the market noise.

## 🚀 Quick Start

```python
import torch
from liquid_layer import LiquidNet

# 1 feature input (e.g., price), 64 hidden ODE neurons, 1 output (probability)
model = LiquidNet(in_features=1, hidden_features=64, out_features=1)

# Input shape: (batch_size, sequence_length, features)
dummy_market_data = torch.randn(32, 60, 1)

# Forward pass (continuous time integration happens inside)
predictions = model(dummy_market_data)
