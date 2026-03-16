#  Prescriptive AI for Energy Arbitrage: Decision-Focused Learning (DFL)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![XGBoost](https://img.shields.io/badge/XGBoost-Predictive%20ML-000000)
![License](https://img.shields.io/badge/License-MIT-green)

## Executive Summary
This repository contains an end-to-end **Prescriptive Artificial Intelligence pipeline** designed to optimize industrial battery arbitrage (e.g., Lithium-Ion Megapacks) in the Day-Ahead European energy market. 

We challenge the standard **Predict-then-Optimize (PFL)** paradigm, demonstrating that minimizing predictive errors (MSE) leads to sub-optimal financial decisions. Instead, we implement a **Decision-Focused Learning (DFL)** architecture in PyTorch, aligning the neural network's loss function directly with the downstream business utility (Financial Regret) while respecting physical degradation constraints (Levelized Cost of Storage - LCOS).

## 🚀 Key Features & Pipeline Architecture

### 1. Data Engineering
* **Data Sources:** Seamless integration of ENTSO-E API (Day-Ahead Prices) and Open-Meteo API (Solar Radiation, Wind Speed) from 2019 to 2025.
* **Cyclical Encoding:** Trigonometric transformations (Sine/Cosine) to preserve the continuous nature of time variables.
* **Leakage Prevention:** Autoregressive lags (24h, 168h) and rolling statistics strictly engineered to prevent Look-Ahead Bias.

### 2. Unsupervised Regime Shift Detection
* **Deep Autoencoders:** Trained on stationary market data (2019-2020) to mathematically isolate the 2021-2022 European Energy Crisis.
* **Anomaly Scoring:** Utilizing Reconstruction Error (MSE) to detect non-linear structural breaks in market dynamics without human bias.

### 3. Explainable AI (XAI) & Predictive Modeling
* **XGBoost Regressor:** Baseline forecasting model evaluated strictly out-of-sample.
* **SHAP Values:** Extracted to decode the physical laws of the power grid, autonomously proving the *Merit Order Effect* (cannibalization of prices by high renewable generation).

### 4. Prescriptive Optimization (DFL)
* **Black-Box Differentiation:** Bypassing the non-differentiable `argmax` operator using a Softmax continuous relaxation inspired by the Score Function Gradient Estimator (SFGE).
* **Industrial Constraints:** Injection of a physical Levelized Cost of Storage (LCOS) penalty of 15 €/MWh. The network learns to remain idle when the daily market spread is unviable.

## 📊 Business Value & Results

Tested on an out-of-sample operational simulation, the DFL architecture vastly outperformed the standard MSE-driven baseline. The PFL model, suffering from regression to the mean, underestimated volatility and remained idle for 19 days. The DFL model accurately pinpointed maximum daily spreads, ensuring 100% Asset Utilization.

| Operational Metric (Net of LCOS) | PFL (MSE Loss) | DFL (Regret Loss) |
| :--- | :--- | :--- |
| **Net Profit (€)** | 98,055 | **110,393** |
| **Financial Advantage (€)** | - | **+ 12,338** |
| **Idle Days (Missed Trades)** | 19 | **0** |
| **Asset Utilization** | 99.2% | **100%** |

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/carmine559/Energy-Arbitrage-DFL.git](https://github.com/carmine559/Energy-Arbitrage-DFL.git)
   cd Energy-Arbitrage-DFL

2. **Create a virtual environment and install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **API Keys:**
   - Obtain an API key for ENTSO-E from https://transparency.entsoe.eu/ and set it as environment variable. Open-Meteo does not require an API key.
     ```bash
     export ENTSOE_API_KEY='your_entsoe_api_key'
     
4. **Run the pipeline:**
    ```bash
    Execute the main.ipynb Jupyter Notebook sequentially to reproduce the data ingestion, training, and financial evaluation.

## Repository Structure
``` 
    ├── images/                 # Saved plots (Autoencoder MSE, SHAP summary, Profit curves)
    ├── main.ipynb              # Core pipeline (Data ingestion, ML, PyTorch DFL)
    ├── report.pdf              # Project report with analysis and results of the DFL architecture
    ├── requirements.txt        # Python dependencies
    └── README.md               # Project documentation

## Authors
* **Carmine Santella** - [Your LinkedIn](https://www.linkedin.com/in/carmine-santella/) | [Your GitHub](https://github.com/carmine559)
* **Tommaso Bernardini** - [Collaborator LinkedIn](https://www.linkedin.com/in/tommaso-bernardini-35a409348/) | [Collaborator GitHub](https://github.com/t0mm4s02)
