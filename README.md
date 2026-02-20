# HVAC Transformer-MPC: Building Energy & Indoor Environmental Quality Forecasting

This repository contains the official implementation of the **Transformer-based Deep Learning** architecture for predicting indoor temperature and optimizing HVAC energy consumption, as described in the paper: 

> **"Energy modeling and predictive control of environmental quality for building energy management using machine learning"** > *Published in Energy for Sustainable Development, Vol. 74 (2023).*

---

## 📖 Overview
Heating, Ventilation, and Air Conditioning (HVAC) systems are responsible for significant global energy consumption. This project implements a high-performance **Transformer-Encoder** model to forecast indoor environmental parameters. By accurately predicting temperature, the system enables **Model Predictive Control (MPC)** to maintain occupant comfort while reducing energy usage by up to **50%**.



## 🏗️ Model Architecture
The implemented architecture deviates from standard RNNs/LSTMs by using self-attention mechanisms to capture long-range dependencies in multivariate time-series data.

- **Look-back Window:** 60 minutes of historical data.
- **Encoder Blocks:** 4 Transformer layers.
- **Attention Heads:** 8 Multi-head attention units.
- **Feed-Forward Network:** 1D Convolutional layers (Kernel size = 1) to maintain spatial-temporal features.
- **Output:** Multi-horizon forecasting (1, 15, 30, and 60 minutes).



## 📁 Repository Structure
```bash
├── data/                   # Dataset files (Environmental sensors, HVAC load)
├── models/
│   ├── transformers.py     # Main Transformer class implementation
│   └── utils.py            # Custom metrics (sMAPE, RMSE) and data processing
├── checkpoints/            # Saved .hdf5 model weights
├── parameters.json         # Model hyperparameters (Look-back, heads, dropout)
├── requirements.txt        # Python dependencies
└── main.ipynb              # Training and evaluation notebook
