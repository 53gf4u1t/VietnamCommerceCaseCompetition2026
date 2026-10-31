# Segment Marketing Budget ML Predictor — Technical Evaluation (`ML_METRICS.md`)

## 1. Model Architecture & Train/Test Split

- **Estimator**: 4 Segment-Specific `HistGradientBoostingRegressor` models (`max_iter=220`, `learning_rate=0.065`, `max_depth=6`, `l2_regularization=1.2`)
- **Training Split (80%)**: `240` Monte Carlo trajectories x `60` months = **`14,400` monthly state transitions**
- **Held-Out Test Split (20%)**: `60` Monte Carlo trajectories x `60` months = **`3,600` monthly state transitions**
- **Input Features (12)**: Last month's active unique customers by segment (4), last month's player visits by segment (4), social media followers (`k`), follower-to-customer conversion rate (`~8.57%`), cumulative pickleball trend index ($I_t$), and active court capacity & Venue 2 fill headroom (`6` vs `20` courts).

## 2. Overall & Per-Segment Out-of-Sample Regression Metrics

| Evaluation Split / Segment Target | $R^2$ Score | RMSE (`M VND`) | MAE (`M VND`) | MAPE (`%`) |
| :--- | :---: | :---: | :---: | :---: |
| **In-Sample Training Set (`N = 14,400`)** | **0.9631** | 0.9323 | 0.6840 | 5.20% |
| **Out-of-Sample Test Set — Overall (`N = 3,600`)** | **0.9555** | **1.0323** | **0.7542** | **5.65%** |
| Out-of-Sample: **Competitive Players Budget** | 0.8588 | 0.9095 | 0.6768 | 5.99% |
| Out-of-Sample: **Social Players Budget** | 0.8717 | 1.0304 | 0.7917 | 4.66% |
| Out-of-Sample: **Trend Explorers Budget** | 0.8733 | 1.4057 | 1.0795 | 6.44% |
| Out-of-Sample: **Families Budget** | 0.8730 | 0.6308 | 0.4690 | 5.51% |

- **Oracle Net Contribution Capture on Held-Out Test Set**: **`99.29%`**
- **Net Contribution Uplift Capture vs. Static 25M Baseline**: **`96.35%`**
