# Decision-Focused Learning for Battery Arbitrage on the IT_NORD Day-Ahead Market

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DFL-EE4C2C)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecast-000000)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end pipeline from raw public data (ENTSO-E prices, Open-Meteo weather)
to a trading policy for a 1 MWh battery on the Italian day-ahead market
(IT_NORD zone). The policy is trained with Decision-Focused Learning (DFL),
which optimises net profit after the Levelized Cost of Storage (LCOS) rather
than the statistical accuracy of the price forecast.

## What the project does

1. **Ingests** day-ahead prices (ENTSO-E) and weather (Open-Meteo) for
   2019–2025, aligned on UTC to avoid DST artefacts.
2. **Builds features** — cyclical encoding of hour and month, autoregressive
   lags (24h, 168h), rolling mean computed on the *lagged* price to prevent
   look-ahead bias.
3. **Detects regime shifts** with an autoencoder trained on the quiet
   2019–2020 market. The 2021–2022 gas crisis falls out of the reconstruction
   error without being labelled.
4. **Forecasts prices** with XGBoost, evaluated chronologically on a
   2024–2025 hold-out. Explainability via SHAP (beeswarm, mean-|SHAP|,
   dependency plots) and Boruta all-relevant feature selection, following
   the methodology of the course notebooks on feature attribution and
   hypothesis testing.
5. **Learns a trading policy** with DFL — a softmax-relaxed regret loss
   with the LCOS folded into the training objective and a small
   anti-collision term so charge and discharge can't land on the same hour.

## Why DFL here

For purely linear costs with uncertain coefficients, a well-specified
predict-then-optimise (PFL) model already converges to the optimum, so
DFL buys little. DFL earns its keep in **two-stage stochastic optimisation**
— problems where the cost is non-linear in the uncertain parameter because
of recourse actions. Battery arbitrage with LCOS is exactly that: the
second-stage decision is *whether to trade at all*. If the expected spread
doesn't cover $C_{deg}$, the optimal action is to stay idle. Folding that
into the training loss is what makes DFL genuinely useful on this problem,
not just philosophically appealing.

## What's honest about the results

Both policies are trained on 2019–2023 daily data and evaluated on the
2024–2025 hold-out. The DFL policy consistently beats the MSE-driven PFL
baseline on cumulative net profit, while trading fewer days (it learns to
skip marginal days where the realised spread would not have covered LCOS).
The exact euro advantage is reproducible with the seed in the notebook but
depends on random initialisation and the assumed $C_{deg}$; the point of
the comparison is the shape of the cumulative-profit curves on unseen days,
not a single headline number. The oracle curve is shown on the same plot
as an upper bound.

The previous version of this project reported a headline profit figure
computed on the *training* days themselves; that was an in-sample number
and is not claimed anymore.

## Repository structure

```
.
├── main_restructured.ipynb          # End-to-end pipeline (ingestion → DFL evaluation)
├── Feature_Selection_Module.ipynb   # Permutation importance, SHAP, Boruta, minimal-set
├── Final_Report.tex                 # LaTeX source for the report
├── images/                          # Plots saved by the notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <this repo>
cd <this repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### API key

ENTSO-E requires a free token (request one at
<https://transparency.entsoe.eu/>). The notebook reads it from an
environment variable — **never hard-code it in the source**:

```bash
export ENTSOE_API_KEY='your_token'
```

Open-Meteo does not require a key.

## How to run

1. Open `main_restructured.ipynb` and run cells top to bottom. It will
   create the `images/` directory, fetch 2019–2025 data, train the
   autoencoder, the XGBoost forecaster, and both PFL / DFL policies, and
   save plots in `images/`.
2. Open `Feature_Selection_Module.ipynb` in the **same kernel** (so the
   `model_xgb`, `X_train`, etc. are already in memory) and run it. It
   will produce the permutation-importance bars, aggregated SHAP, the
   Boruta report, and the minimal-set RMSE curve.
3. Read `Final_Report.pdf` for the written narrative.

## Known limitations

- **Minimal battery model.** One cycle per day, no state of charge, no
  efficiency losses, no ramp constraints. A production stack would be a
  MILP; the DFL principle carries over but the training loop would need a
  differentiable solver or a black-box surrogate.
- **Single bidding zone.** IT_NORD only. No cross-zonal arbitrage.
- **Scalar LCOS.** Fixed at 15 €/MWh, consistent with published figures for
  lithium-ion Megapacks. A sensitivity sweep on $C_{deg}$ is a natural
  next step.
- **Day-ahead only.** The intra-day and balancing markets, often more
  volatile than day-ahead, are out of scope.

## Authors

- **Carmine Santella** — [LinkedIn](https://www.linkedin.com/in/carmine-santella/) · [GitHub](https://github.com/carmine559)
- **Tommaso Bernardini** — [LinkedIn](https://www.linkedin.com/in/tommaso-bernardini-35a409348/) · [GitHub](https://github.com/t0mm4s02)

## License

MIT.