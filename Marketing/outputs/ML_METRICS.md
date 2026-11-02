# Segment Marketing Budget ML Predictor - Technical Evaluation (`ML_METRICS.md`)

## 1. Model Architecture & Train/Test Split

- **Estimator**: 4 Segment-Specific `HistGradientBoostingRegressor` models (`max_iter=220`, `learning_rate=0.065`, `max_depth=6`, `l2_regularization=1.2`)
- **Training Split (80%)**: $240\text{ Monte Carlo trajectories} \times 60\text{ months} =$ **`14,400` monthly state transitions**
- **Held-Out Test Split (20%)**: $60\text{ Monte Carlo trajectories} \times 60\text{ months} =$ **`3,600` monthly state transitions**
- **Input Features (12)**: Last month's active unique customers by segment (4), last month's player visits by segment (4), social media followers (`k`), follower-to-customer conversion rate (`~8.57%`), cumulative pickleball trend index ($I_t$), and active court capacity & Venue 2 fill headroom (`6` vs `20` courts).

## 2. Overall & Per-Segment Out-of-Sample Regression Metrics

| Evaluation Split / Segment Target | $R^2$ Score | RMSE (`M VND`) | MAE (`M VND`) | MAPE (`%`) |
| :--- | :---: | :---: | :---: | :---: |
| **In-Sample Training Set (`N = 14,400`)** | **0.9637** | 0.9252 | 0.6769 | 5.14% |
| **Out-of-Sample Test Set - Overall (`N = 3,600`)** | **0.9556** | **1.0307** | **0.7513** | **5.62%** |
| Out-of-Sample: **Competitive Players Budget** | 0.8584 | 0.9108 | 0.6764 | 5.98% |
| Out-of-Sample: **Social Players Budget** | 0.8726 | 1.0266 | 0.7878 | 4.64% |
| Out-of-Sample: **Trend Explorers Budget** | 0.8736 | 1.4042 | 1.0776 | 6.42% |
| Out-of-Sample: **Families Budget** | 0.8742 | 0.6278 | 0.4636 | 5.45% |

- **Oracle Net Contribution Capture on Held-Out Test Set**: **`99.29%`**
- **Net Contribution Uplift Capture vs. Static 25M Baseline**: **`96.36%`**
