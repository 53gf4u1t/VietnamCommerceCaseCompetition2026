# PICK! Adaptive Court Pricing - Simulation & ML Optimization Report
**Dynamic Court Pricing Engine Across Active Venue Capacity (6 Courts M1-M13 -> 20 Courts M14-M60)**

---

## 1. Executive Summary & Case Reconciliation

PICK! operates **6 leased courts** in Months 1-13 (`2,880` available court-hours/month) and scales to **20 courts** in Months 14-60 (`9,600` available court-hours/month) open **16 hours per day (06:00-22:00)** across 30-day months (22 weekdays and 8 weekend days). Under the baseline two-tier static pricing (**160k VND** weekday daytime vs. **260k VND** weekday evening and weekends), PICK! suffers from severe off-peak underuse: **60% of weekday daytime courts sit empty** (40% utilization), while weekday evenings (95% utilization) and weekends (90% utilization) are near capacity.

### Mathematical Reconciliation of Baseline Court Data
Before running stochastic simulations, we verify that the Case Document figures reconcile per 6-court unit (`2,880` available court-hours/month) and scale proportionally across the 20-court network (`9,600` available court-hours/month):
- **Weekday Daytime (06:00-17:00, 11 hrs/day x 22 days = 242 operating hours/month):** At 40% utilization, $1,452 \times 0.40 = 580.8$ booked court-hours on 6 courts ($1,936.0$ on 20 courts) $\times 160\text{k VND} = 92.93\text{M VND}$ ($309.76\text{M VND}$ across 20 courts).
- **Weekday Evening (17:00-22:00, 5 hrs/day x 22 days = 110 operating hours/month):** At 95% utilization, $660 \times 0.95 = 627.0$ booked court-hours on 6 courts ($2,090.0$ on 20 courts) $\times 260\text{k VND} = 163.02\text{M VND}$ ($543.40\text{M VND}$ across 20 courts).
- **Weekends (06:00-22:00, 16 hrs/day x 8 days = 128 operating hours/month):** At 90% utilization, $768 \times 0.90 = 691.2$ booked court-hours on 6 courts ($2,304.0$ on 20 courts) $\times 260\text{k VND} = 179.71\text{M VND}$ ($599.04\text{M VND}$ across 20 courts).
- **Total Monthly Court Hours Booked:** $580.8 + 627.0 + 691.2 = \mathbf{1,899.0\text{ hours}}$ per 6 courts ($\mathbf{6,330.0\text{ hours}}$ at 20-court steady state).
- **Total Monthly Court Hire Revenue:** $92.93 + 163.02 + 179.71 = \mathbf{435.66\text{M VND}}$ per 6 courts ($\mathbf{1,452.20\text{M VND}}$ at 20-court steady state).

### Impact of Machine Learning Adaptive Pricing
| Metric | Baseline Static (6 Cts / 20 Cts) | ML Adaptive Pricing (6 Cts / 20 Cts) | Absolute Change (6 Cts / 20 Cts) | Relative Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Monthly Court Hire Revenue** | **436.86M / 1,456.20M VND** | **460.75M / 1,535.84M VND** | **+23.89M / +79.64M VND/mo** | **+5.47%** |
| **Annualized Court Hire Revenue** | 5242.3M / 17,474.4M VND/yr | 5529.0M / 18,430.1M VND/yr | +286.7M / +955.7M VND/yr | +5.47% |
| **Monthly Booked Court-Hours** | 1905 / 6,351 hrs | 2191 / 7,302 hrs | +285 / +951 hrs/mo | +15.0% |
| **Weekday Daytime Utilization** | **40.3%** (2.40 / 8.00 cts) | **66.6%** (3.86 / 12.87 cts) | **+26.3 pp** | **+65.3%** |
| **Weekday Evening Utilization** | 94.9% | 87.9% | -7.0 pp | Balanced Peak |
| **Weekend Utilization** | 90.4% | 83.8% | -6.6 pp | Midday Filled |
| **Ancillary Net Margin Lift** | Baseline Included | +9.01M / +30.03M VND/mo | +108.1M / +360.3M VND/yr | High-Margin Pull |
| **Total Monthly Operating Profit Contribution** | 160.00M VND (Venue 1) | **192.09M VND** (Venue 1) | **+32.09M / +106.96M VND/mo** | **+20.1%** |
| **Downside Demand Stress Court Revenue** | 305.82M / 1,019.41M VND | **314.46M / 1,048.20M VND** | **+8.64M / +28.79M VND/mo** | **+2.8%** |

---

## 2. Configurable Assumptions & Statistical Distributions

All parameters are stored cleanly in `configs/pricing.toml` and validated in `01_assumptions_distributions.png`:

1. **Monthly Court-Hours Booked Distribution:** Emerges naturally from hourly court Bernoulli trials across 300 simulated months, yielding a Normal distribution with $\mu = 1901.9$ hours and $\sigma = 54.0$ hours per 6-court venue ($6,330.0$ hours across 20 courts).
2. **Stochastic Regime, Daily & Hourly Distributions:**
   - **Weekday Daytime Utilization:** $\mathcal{N}(\mu = 40.0\%, \sigma = 3.2\%)$, clipped to $[28\%, 52\%]$.
   - **Weekday Evening Utilization:** $\mathcal{N}(\mu = 95.0\%, \sigma = 1.6\%)$, clipped to $[88\%, 99\%]$.
   - **Weekend Utilization:** $\mathcal{N}(\mu = 90.0\%, \sigma = 2.2\%)$, clipped to $[82\%, 97\%]$.
   - **Daily Random Demand Multiplier (`[daily_random_multiplier]`):** Sampled independently for every day from $\mathcal{N}(\mu = 1.00, \sigma = 0.055)$, clipped to $[0.82, 1.18]$ (simulated mean $\mu = 1.000 \pm 0.055$ across 9,000 days).
   - **Hourly Random Demand Shock (`[hourly_random_shock]`):** Sampled independently for every hour from $\mathcal{N}(\mu = 1.00, \sigma = 0.045)$, clipped to $[0.85, 1.15]$ (simulated mean $\mu = 1.000 \pm 0.045$ across 144,000 court-hours).
3. **Regime Linear Demand Curves Constrained to Empirical Anchor Points:**
   - We model **3 separate linear demand curves** corresponding to PICK!'s 3 operating regimes, expressed in **court utilization $u_r(x) \in [0\%, 100\%]$** (and scaled by active court capacity $C_t \in \{6, 20\}$ so $Q_r(x, C_t) = C_t \cdot u_r(x)$):

| Operating Regime $r$ | Known Anchor Price $x_0$ | Known Anchor Utilization $u_0$ (`6 Cts / 20 Cts`) | Simulated Mean @ $x_0$ | Normalized Utilization Demand Curve $u_r(x)$ | Slope Distribution $a_r$ ($\mu \pm \sigma$) | Intercept Distribution $b_r$ ($\mu \pm \sigma$) | ML Adaptive Mean Price $\hat{x}$ | ML Expected Utilization (`6 Cts / 20 Cts`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Weekday Daytime** | **160k VND** | **40.0%** (`2.40 / 8.00 cts`) | **40.1%** | $u_1(x) = \text{clip}(-0.583\% \cdot x + 133.5\%, 0, 100\%)$ | `-0.0350 +/- 0.0032` | `8.01 +/- 0.59` | **115k VND** | **66.6%** (`4.00 / 13.32 cts`) |
| **2. Weekday Evening** | **260k VND** | **95.0%** (`5.70 / 19.00 cts`) | **95.0%** | $u_2(x) = \text{clip}(-0.233\% \cdot x + 155.7\%, 0, 100\%)$ | `-0.0140 +/- 0.0018` | `9.34 +/- 0.49` | **290k VND** | **87.9%** (`5.27 / 17.57 cts`) |
| **3. Weekend** | **260k VND** | **90.0%** (`5.40 / 18.00 cts`) | **90.0%** | $u_3(x) = \text{clip}(-0.283\% \cdot x + 163.6\%, 0, 100\%)$ | `-0.0170 +/- 0.0022` | `9.81 +/- 0.60` | **283k VND** | **83.8%** (`5.03 / 16.76 cts`) |

   - **Why Regime 1 (Weekday Daytime) Lowers Price:** At the anchor point $(160\text{k}, 40.0\%\text{ utilization})$, university students and young social players are price-elastic ($|\varepsilon| = 2.33 > 1$). Lowering the daytime price toward $\approx 115\text{k VND}$ **boosts daytime court utilization from 40.0% to 66.6%** (`2.40 -> 3.86` courts in Phase 1; `8.00 -> 12.87` courts in Phase 2) and **increases daytime hourly revenue by +18.9%**.
   - **Why Regime 2 (Weekday Evening) & Regime 3 (Weekend) Adjust Around Peak:** At the anchor points $(260\text{k}, 95.0\%)$ and $(260\text{k}, 90.0\%)$, after-work and weekend players are price-inelastic ($|\varepsilon| < 1$). The ML model nudges prime peak hours to **290k-283k VND** (while capping the 21:00-22:00 hour at $\le 250\text{k VND}$), capturing incremental revenue without sacrificing high occupancy.
4. **Trust, Affordability & Noise Guardrails:**
   - Prices are bounded within **[85k VND, 300k VND]** and rounded to clean **5k VND increments** so prices remain transparent and predictable on the PICK! custom booking platform.
   - **Late-Evening Noise Management Cap:** Capped at **$\le 250\text{k VND}$** during `21:00-22:00` to incentivize price-sensitive groups to book earlier slots and respect ward noise rules after 21:00.

---

## 3. Machine Learning Architecture & Pricing Mechanics

- **Feature Vector:** The ML model receives real-time court occupancy indicators alongside calendar context (`hour_of_day`, `day_of_week`, `day_of_month`).
- **Training Objective:** On the 240 training months ($115,200$ court-hour observations), a regularized **Histogram Gradient Boosting Regressor** (`HistGradientBoostingRegressor`) learns the posterior expected optimal price $\hat{x} = f(\mathbf{x}_{\text{input}})$ solely from observable booking states and calendar context, achieving **99.8% of the perfect-foresight Oracle revenue** across both the 6-court and 20-court capacity phases.

---

## 4. Visual Package Guide (`AdaptivePricing/outputs/`)

1. **`01_assumptions_distributions.png`**: Diagnostic view of all configured probability distributions (Monthly Booked Hours, Regime Utilization Normals, Regime Slopes, Daily Random Multiplier, Hourly Random Shock, and Regime Intercepts).
2. **`02_demand_curves_variations.png`**: Regime linear demand curves expressed in **Court Utilization (`0%-100%`)** with dual **6-Court / 20-Court** capacity scaling (top row) and quadratic hourly revenue frontiers (bottom row).
3. **`03_daily_revenue_comparison.png`**: 30-day revenue comparison between Baseline Static and ML Adaptive pricing with cumulative monthly revenue trajectory and dual 6-court / 20-court scaling.
4. **`04_monthly_calendar_heatmap.png`**: Consolidated **2-Row Hourly Pricing Schedule (`Weekday` vs. `Weekend Day` across `06:00-22:00`)** paired with hourly utilization comparison.
5. **`05_occupancy_and_stress_test.png`**: Regime utilization comparison and downside demand stress test across active venue capacity.
6. **`06_ml_model_diagnostics.png`**: Out-of-sample ML validation suite (Parity plot, residual distribution, permutation feature importances, and boosting learning curve).
