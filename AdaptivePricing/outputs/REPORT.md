# PICK! Adaptive Court Pricing — Simulation & ML Optimization Report
**Variant of Path C: Custom Community & Booking Platform | Plan Start: Jan 2027 (FY 2027–FY 2031)**

---

## 1. Executive Summary & Case Reconciliation

PICK! currently operates **6 leased courts** in Ho Chi Minh City open **16 hours per day (06:00–22:00)**, totaling **2,880 available court-hours per 30-day month** (22 weekdays and 8 weekend days). Under the current two-tier static pricing (**160k VND** weekday daytime vs. **260k VND** weekday evening and weekends), PICK! suffers from severe off-peak underuse: **60% of weekday daytime courts sit empty** (40% utilization), while weekday evenings (95% utilization) and weekends (90% utilization) are near capacity.

### Exact Mathematical Reconciliation of Case Exhibit 1 Data
Before running stochastic simulations, we verify that the Case Document figures reconcile exactly:
- **Weekday Daytime (06:00–17:00, 11 hrs/day × 22 days = 1,452 court-hours):** At 40% utilization, $1,452 \times 0.40 = 580.8$ booked hours $\times 160\text{k VND} = 92.93\text{M VND}$.
- **Weekday Evening (17:00–22:00, 5 hrs/day × 22 days = 660 court-hours):** At 95% utilization, $660 \times 0.95 = 627.0$ booked hours $\times 260\text{k VND} = 163.02\text{M VND}$.
- **Weekends (06:00–22:00, 16 hrs/day × 8 days = 768 court-hours):** At 90% utilization, $768 \times 0.90 = 691.2$ booked hours $\times 260\text{k VND} = 179.71\text{M VND}$.
- **Total Monthly Court Hours Booked:** $580.8 + 627.0 + 691.2 = \mathbf{1,899.0\text{ hours}}$ (Case Exhibit 1 reports *"About 1,900"*).
- **Total Monthly Court Hire Revenue:** $92.93 + 163.02 + 179.71 = \mathbf{435.66\text{M VND}}$ (Case Exhibit 1 reports *"VND 435 million"*, average $\sim 229.4\text{k VND/hour}$).

### Headline Impact of ML Adaptive Pricing
| Metric | Baseline Static (160k / 260k) | ML Adaptive Pricing (85k–300k) | Absolute Change | Relative Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Monthly Court Hire Revenue** | **436.86M VND** | **460.75M VND** | **+23.89M VND/mo** | **+5.47%** |
| **Annualized Court Hire Revenue** | 5242.3M VND/yr | 5529.0M VND/yr | +286.7M VND/yr | +5.47% |
| **Monthly Booked Court-Hours** | 1905 hrs / 2,880 | 2191 hrs / 2,880 | +285 hrs/mo | +15.0% |
| **Weekday Daytime Utilization (06–17h)** | **40.3%** (2.40 courts) | **66.6%** (3.86 courts) | **+26.3 pp** | **+65.3%** |
| **Weekday Evening Utilization (17–22h)** | 94.9% | 87.9% | -7.0 pp | Balanced Peak |
| **Weekend Utilization (Sat–Sun 06–22h)** | 90.4% | 83.8% | -6.6 pp | Midday Filled |
| **Ancillary Net Margin Lift (F&B + Rentals)** | Baseline Included | +9.01M VND/mo | +108.1M VND/yr | High-Margin Pull |
| **Total Monthly Operating Profit Contribution** | 160.00M VND | **192.09M VND** | **+32.09M VND/mo** | **+20.1%** |
| **-30% Demand Shock Court Revenue** | 305.82M VND | **314.46M VND** | **+8.64M VND/mo** | **+2.8%** |

---

## 2. Configurable Assumptions & Statistical Distributions

All parameters are stored cleanly in `AdaptivePricing/config.toml` and validated in `01_assumptions_distributions.png`:

1. **Monthly Court-Hours Booked Distribution:** Emerges naturally from the sum of 2,880 hourly court Bernoulli trials across 300 simulated months, yielding a Normal distribution with $\mu = 1901.9$ hours and $\sigma = 54.0$ hours (matching Case Exhibit 1's $1,900$ hours).
2. **Purely Stochastic Regime, Daily & Hourly Distributions (Zero Hardcoded Arrays):**
   - **Weekday Daytime Utilization (06:00–17:00):** $\mathcal{N}(\mu = 40.0\%, \sigma = 3.2\%)$, clipped to $[28\%, 52\%]$.
   - **Weekday Evening Utilization (17:00–22:00):** $\mathcal{N}(\mu = 95.0\%, \sigma = 1.6\%)$, clipped to $[88\%, 99\%]$.
   - **Weekend Utilization (06:00–22:00):** $\mathcal{N}(\mu = 90.0\%, \sigma = 2.2\%)$, clipped to $[82\%, 97\%]$.
   - **Daily Random Demand Multiplier (`[daily_random_multiplier]`):** Sampled independently for every day from $\mathcal{N}(\mu = 1.00, \sigma = 0.055)$, clipped to $[0.82, 1.18]$ (simulated mean $\mu = 1.000 \pm 0.055$ across 9,000 days).
   - **Hourly Random Demand Shock (`[hourly_random_shock]`):** Sampled independently for every hour from $\mathcal{N}(\mu = 1.00, \sigma = 0.045)$, clipped to $[0.85, 1.15]$ (simulated mean $\mu = 1.000 \pm 0.045$ across 144,000 court-hours).
3. **Three Separate Regime Linear Demand Curves $Q_r(x) = \text{clip}(a_r x + b_r, 0, 6)$ Constrained to Known Empirical Anchor Points:**
   - Rather than fitting a single generic demand curve, we model **3 separate linear demand curves** corresponding to PICK!'s 3 operating regimes, each mathematically constrained so that at the current static price $x_0 \in \{160\text{k}, 260\text{k}\}\text{ VND}$, the curve passes directly through the known Case Exhibit 1 court booking density $Q_0 = 6 \times u_0$:
     $$b_r = Q_{0, r} - a_r x_{0, r}$$

| Operating Regime $r$ | Known Anchor Price $x_0$ | Known Anchor Booking Density $u_0$ ($Q_0$) | Simulated Mean @ $x_0$ | Mean Linear Demand Formula $Q_r(x) = \text{clip}(a_r x + b_r, 0, 6)$ | Slope Distribution $a_r$ ($\mu \pm \sigma$) | Intercept Distribution $b_r$ ($\mu \pm \sigma$) | ML Adaptive Mean Price $\hat{x}$ | ML Expected Utilization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Weekday Daytime (06:00–17:00)** | **160k VND** | **40.0%** (`2.40` / 6 cts) | **40.1%** (`2.41` cts) | $Q_1(x) = \text{clip}(-0.0350x + 8.01, 0, 6)$ | `-0.0350 ± 0.0032` | `8.01 ± 0.59` | **115k VND** | **66.6%** (`4.00` cts) |
| **2. Weekday Evening (17:00–22:00)** | **260k VND** | **95.0%** (`5.70` / 6 cts) | **95.0%** (`5.70` cts) | $Q_2(x) = \text{clip}(-0.0140x + 9.34, 0, 6)$ | `-0.0140 ± 0.0018` | `9.34 ± 0.49` | **290k VND** | **87.9%** (`5.27` cts) |
| **3. Weekend (Sat–Sun 06:00–22:00)** | **260k VND** | **90.0%** (`5.40` / 6 cts) | **90.0%** (`5.40` cts) | $Q_3(x) = \text{clip}(-0.0170x + 9.81, 0, 6)$ | `-0.0170 ± 0.0022` | `9.81 ± 0.60` | **283k VND** | **83.8%** (`5.03` cts) |

   - **Why Regime 1 (Weekday Daytime) Drops Price from 160k to ~115k VND:** At the anchor point $(160\text{k}, 2.40\text{ courts})$, university students and young social players (aged 16–30) are highly price-elastic ($a_1 = -0.035$ courts per k VND, meaning point elasticity $|\varepsilon| = |a_1 \cdot (160 / 2.40)| = 2.33 > 1$). Because marginal revenue is negative at 160k VND, lowering the daytime price toward the vertex $x_1^* = -b_1/(2a_1) \approx 115\text{k VND}$ simultaneously **boosts daytime court utilization from 40.0% to 66.6%** and **increases daytime hourly revenue by +19%**.
   - **Why Regime 2 (Weekday Evening) & Regime 3 (Weekend) Adjust Around 260k VND:** At the anchor points $(260\text{k}, 5.70\text{ courts [95\%]})$ and $(260\text{k}, 5.40\text{ courts [90\%]})$, after-work and weekend players are relatively price-inelastic ($|a_2| \approx 0.014$, $|a_3| \approx 0.017$, so $|\varepsilon| < 1$). The ML model nudges prime peak hours to **290k–283k VND** (while capping the 21:00–22:00 noise-complaint hour at $\le 250\text{k VND}$), capturing incremental revenue without sacrificing high occupancy.
4. **Trust, Affordability & Noise Guardrails:**
   - Prices are bounded within **[85k VND, 300k VND]** and rounded to clean **5k VND increments** so prices remain transparent and predictable on the PICK! custom booking platform.
   - **21:00–22:00 Noise Management Cap:** Capped at **$\le 250\text{k VND}$** (below the 285k–300k prime evening rate) to incentivize price-sensitive groups to book earlier afternoon/evening slots and respect ward office noise rules after 21:00.

---

## 3. Machine Learning Architecture & Pricing Mechanics

- **Feature Vector (9 Inputs per Court-Hour):** Exactly as specified, the ML model receives:
  $$\mathbf{x}_{\text{input}} = [c_1, c_2, c_3, c_4, c_5, c_6, \text{hour\_of\_day}, \text{day\_of\_week}, \text{day\_of\_month}]$$
  where $c_1, \dots, c_6 \in \{0, 1\}$ represent the binary rental state of each of PICK!'s 6 courts, `hour_of_day` $\in \{6, \dots, 21\}$, `day_of_week` $\in \{0, \dots, 6\}$, and `day_of_month` $\in \{1, \dots, 30\}$.
- **Training Objective:** On the 240 training months ($115,200$ court-hour instances), for each simulated trajectory with underlying parameters $(a, b)$, the exact revenue-maximizing price on the capacity-capped linear curve $R(x) = x \cdot \text{clip}(a x + b, 0, 6)$ is:
  $$x^* = \text{clip}\left(\max\left(-\frac{b}{2a}, \frac{6 - b}{a}\right), x_{\min}, x_{\max}\right)$$
  A regularized **Histogram Gradient Boosting Regressor** (`HistGradientBoostingRegressor`) learns the posterior expected optimal price $\hat{x} = f(\mathbf{x}_{\text{input}})$ solely from observable booking states and calendar context, achieving **99.8% of the perfect-foresight Oracle revenue**.

---

## 4. Visual Package Guide (`AdaptivePricing/outputs/`)

1. **`01_assumptions_distributions.png`**: 8-panel ($2 \times 4$) diagnostic view of all configured probability distributions (Monthly Booked Hours, Weekday Daytime/Evening/Weekend Utilization Normals, 3 Regime Slopes $a_r$, Daily Random Multiplier $m_d$, Hourly Random Shock $s_{d,h}$, and 3 Regime Intercepts $b_r$).
2. **`02_demand_curves_variations.png`**: 6-panel ($2 \times 3$) view of the 3 separate regime linear demand curves $Q_r(x) = \text{clip}(a_r x + b_r, 0, 6)$ anchored at `(160k, 40.0%)`, `(260k, 95.0%)`, and `(260k, 90.0%)` (top row) and their 3 quadratic revenue frontiers $R_r(x) = x Q_r(x)$ (bottom row).
3. **`03_daily_revenue_comparison.png`**: 30-day double-bar chart comparing Baseline vs. ML Adaptive daily revenue (with $\pm 1\text{ SD}$ bars and weekend highlights) plus cumulative monthly revenue trajectory.
4. **`04_monthly_calendar_heatmap.png`**: Full 30-day calendar view where **each day contains a $4 \times 4$ hourly grid heatmap** for all 16 hours (06:00–22:00), displaying the exact recommended price (in k VND) for every hour of the month.
5. **`05_occupancy_and_stress_test.png`**: Demonstrates how ML pricing lifts weekday daytime utilization from **40.0% to 66.6%** and protects PICK!'s profitability under the **-30% lower-demand scenario**.
6. **`06_ml_model_diagnostics.png`**: 4-panel technical ML validation suite (Out-of-sample parity plot, residual distribution, permutation feature importances, and boosting learning curve).
