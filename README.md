# 2026 Vietnam Commerce Case Competition — PICK! Strategic & Financial Plan (2027–2031)

> **AI Usage Acknowledgement**  
> All core strategic decisions, financial modeling architecture, and quantitative assumptions in this repository were conceived and directed by our team—including the 4-stage execution sequence, the two-tranche equity structure (`500M VND` upon signing + `7,500M VND` at Q4 2027 expansion), the fully stochastic distribution-based simulation design (`configs/*.toml`), the `8.57%` follower-to-customer funnel, and the 12 customer-segment ancillary spending distributions. We used **OpenAI Codex** and **Google Antigravity** to accelerate Python implementation (`NumPy`, `SciPy`, `scikit-learn`), generate `Matplotlib` visualizations, cross-check software/construction cost benchmarks, and format documentation. Every assumption, distribution parameter, and financial output was independently audited and verified by the team.

---

## 1. Strategic Overview & 5-Year Financial Results

PICK! has `42,000` social followers and `90%–95%` evening court occupancy, but **`60%` of weekday daytime courts sit empty** and casual visitors churn once viral excitement fades. Expanding venues immediately would multiply daytime vacancy, while launching an e-commerce paddle brand requires manufacturing and distribution capabilities PICK! does not have (`18%` survey intent).

Instead, we execute a **4-stage operational and capital sequence** within the **`8,000.0M VND` investment ceiling**:

| Stage | Timeline | Action | Why It Works |
| :--- | :--- | :--- | :--- |
| **1. Dynamic Pricing & Cash Buffer** | `Months 1–4` (`Jan–Apr 2027`) | Deploy ML time-of-day court pricing on Venue 1 (`6` courts); build `PICK! Social` in-house (`296.95M VND`); draw **`+500.0M VND` (Tranche 1)** into reserve upon signing. | Cuts midday `10:00–14:00` prices (`160k -> 105k VND/hr`) to fill empty daytime slots (`18.3% -> 46.8%` occupancy), lifting revenue **`+24.92M VND/mo`** immediately while keeping cash `>= 1,000.73M VND`. |
| **2. `PICK! Social` & Segment Marketing** | `Months 5–9` (`May–Sep 2027`) | Launch `PICK! Social` (`399,000 VND/mo` membership, skill matchmaking, leagues) and ML segment marketing across `5` full operating months. | Converts casual visitors into **`1,422` paying subscribers** and builds cash reserves to **`+2,438.49M VND`** before committing to a second venue lease. |
| **3. Q4 2027 14-Court Expansion** | `Months 10–13` (`Oct 2027–Jan 2028`) | Draw **`+7,500.0M VND` (Tranche 2)** (`7,000M` fit-out + `500M` reserve); execute `16-week` turnkey build of Venue 2 (`+14` courts, `-1,750M VND/mo` leveled CapEx). | Funded when Venue 1 + `PICK! Social` is already generating `+297M–435M VND/mo` in operating profit. |
| **4. 20-Court Network Scale** | `Months 14–60` (`Feb 2028–Dec 2031`) | Open Venue 2 in `Feb 2028` (`20` courts total); pre-built `PICK! Social` community drives 6-to-9 month utilization ramp (`49.9% -> 69.9%`). | Reaches **`20,473.28M VND` Year 3 operating profit** (**`5.85x`** the `3,500M VND` investor target; **`100%` simulation pass rate**). |

![Revenue, Costs, and Operating Profit: Our Plan vs. Stochastic Baseline](FinancialModel/outputs/04_revenue_costs_profit_comparison_charts.png)

![5-Year Annual Financial Statement Projection (FY 2027 - FY 2031)](FinancialModel/outputs/03_fs_annual_5y_projection_sheet.png)

### 5-Year Performance Summary (`N = 300` Monte Carlo Trajectories)

| Metric | Stochastic Baseline (`6 Courts`) | Our Plan (`Expected Mean`) | Our Plan (`90% CI: P05–P95`) | Permanent `-30%` Demand Stress |
| :--- | :---: | :---: | :---: | :---: |
| **Year 1 (`FY 2027`) EBITDA** | `1,856.4M VND` | **`2,654.21M VND`** | `[2,226M, 3,098M]` | `-265.10M VND` |
| **Year 2 (`FY 2028`) EBITDA** | `1,826.2M VND` | **`13,134.12M VND`** | `[12,240M, 14,009M]` | `+5,259.40M VND` |
| **Year 3 (`FY 2029`) EBITDA (`Target: 3,500M`)** | `1,753.0M VND` | **`20,473.28M VND` (`5.85x`)** | `[19,388M, 21,542M]` (`100% Pass`) | **`+10,285.59M VND` (`2.94x`)** |
| **Year 4 (`FY 2030`) EBITDA** | `1,794.1M VND` | **`22,633.72M VND`** | `[21,495M, 23,777M]` | `+11,761.20M VND` |
| **Year 5 (`FY 2031`) EBITDA** | `1,664.5M VND` | **`24,205.08M VND`** | `[22,953M, 25,388M]` | `+12,817.00M VND` |
| **5Y Cumulative Revenue** | `37,883.4M VND` | **`164,066.74M VND`** | `[161,012M, 167,118M]` | `120,035.40M VND` |
| **5Y Cumulative EBITDA** | `8,894.2M VND` | **`83,100.41M VND` (`50.65%`)** | `[80,046M, 86,152M]` | `39,857.00M VND` |
| **Minimum Monthly Cash (`M1–M60`)** | `+646.97M VND` | **`+1,000.73M VND` (`M3`)** | `P05 Min: +954.57M VND` | **`+217.59M VND` (`M6`)** |
| **Month 60 Ending Cash** | `9,394.21M VND` | **`84,600.41M VND`** | `[81,546M, 87,652M]` | `41,357.00M VND` |

---

## 2. How Do You Trust Our Inputs? (Grounded in Case Data & Cited Benchmarks)

Every input is modeled as a **bounded probability distribution (`Truncated Normal`)** in [`configs/`](configs) and reconciles directly to the Case Data Pack:

### 2.1 Hourly Court Utilization & Price Elasticity ([`AdaptivePricing/`](AdaptivePricing))
- Grounded in the `6`-court `06:00–22:00` schedule (`435.2M VND/mo` baseline court revenue, `40%` weekday daytime vs. `95%` evening / `90%` weekend occupancy).

![Hourly Court Utilization & Dynamic Pricing Regimes](AdaptivePricing/outputs/01_hourly_utilization_and_pricing_regimes.png)

![Price-Demand Elasticity Curves by Time Regime](AdaptivePricing/outputs/02_elasticity_and_demand_response_curves.png)

### 2.2 Customer Funnel & 12 Segment-by-Ancillary Spending Distributions ([`Marketing/`](Marketing))
- **Explicit Funnel Assumption**: Our `3,600` monthly unique customers come from our `42,000` TikTok/Instagram followers (`3,600 / 42,000 = 8.571%` conversion, modeled as `N(8.571%, 0.85%)`), generating `7,596` baseline player visits across 4 customer segments.
- **Exact VND Marketing Cost per Visit**: Mapping the `25.0M VND/mo` baseline marketing spend across segments yields **`x = 1,543 VND/visit`** (Competitive), **`y = 3,373 VND/visit`** (Social), **`z = 6,313 VND/visit`** (Trend Explorers), and **`w = 6,173 VND/visit`** (Families).
- **12 Ancillary Spending Distributions (`4 Segments x 3 Lines`)**: Each segment has its own distribution for **Paddle/Ball Rental**, **Coaching**, and **Food & Drinks**. Multiplied by baseline visits, they sum to **`204.80M VND/month`**—matching the `32%` non-court revenue in the Data Pack exactly.

![Marketing Funnel & 12 Segment-by-Ancillary Spending Distributions](Marketing/outputs/01_marketing_funnel_and_segment_response_distributions.png)

### 2.3 Cited Engineering, Cloud & Construction Costs
- **`PICK! Social` (`296.95M VND` CapEx + `26.50M VND/mo` OpEx + `1.50%` Payment Gateway COGS)**: Scoped across `21` work packages (`244.40M` base + `21.5%` buffer) using *COCOMO II*, *PMI PMBOK 7th Ed.*, *ITviec/TopDev Vietnam Tech Salary Reports*, and *NAPAS/VietQR + Momo/VNPay fee schedules* ([`PickSocialProjectTimeline/`](PickSocialProjectTimeline)).
- **Venue 2 Fit-Out (`7,000.0M VND` CapEx + `330.0M VND/mo` Lease + Scaled Utilities/Maint/Staff)**: Scoped across `18` construction packages (`5,950.0M` base + `1,050.0M` / `17.65%` buffer) using *Vietnam Ministry of Construction Circular 11/2021/TT-BXD* ([`VenueScalingTimeline/`](VenueScalingTimeline)); when the 14 new courts open in `Month 14`, electricity/water (`+82.0M/mo`), maintenance/insurance (`+38.0M/mo`), and staff wages (`+215.0M/mo`) scale up automatically alongside variable COGS.

---

## 3. How Do You Trust Our Results? (Stochastic Rigor, Out-of-Sample ML & Stress Testing)

Static spreadsheets fail because they ignore variance, capacity ceilings, and external shocks. We verify our results through **four mechanisms**:

### 3.1 Fair Baseline Comparison Under Identical Simulation Shocks
- The **Case Baseline (`6` courts, static pricing, `25M VND/mo` marketing)** runs through the **exact same `N = 300` simulation lines and monthly random draws** as our plan—fluctuating around `628.9M–633.7M VND/mo` revenue and `1,664.5M–1,856.4M VND/yr` profit (`0.0%` chance of hitting `3,500M VND`).

### 3.2 Out-of-Sample Machine Learning Validation (`80/20` Train/Test Split)
- **`AdaptivePricing` ML (`HistGradientBoostingRegressor`)**: Held-out **Test `R^2 = 0.9922`** (`MAE = 2.15%`), capturing **`98.8%`** of Oracle revenue lift (`+24.92M VND/mo` per `6` courts).
- **`Marketing` Segment Allocator (`MultiOutputRegressor`)**: Predicts Month `t`'s 4-segment marketing budget from **Month `t-1`'s realized segment metrics** (zero look-ahead bias). Held-out **Test `R^2 = 0.9555`**, capturing **`99.29%`** of Oracle profit and lifting ancillary revenue from **`204.8M -> 500.05M VND/mo`** (`5Y avg`).

![Adaptive Pricing Out-of-Sample ML Validation & Revenue Lift](AdaptivePricing/outputs/03_ml_validation_and_revenue_lift.png)

![ML Segment Marketing Budget & Visits Trajectory vs. Stochastic Baseline](Marketing/outputs/02_ml_segment_marketing_budget_and_visits_trajectory.png)

![Segment Ancillary Revenue Breakdown: Paddle/Balls, Coaching, and F&B](Marketing/outputs/03_segment_ancillary_revenue_breakdown.png)

![Marketing ML Out-of-Sample Diagnostics & Feature Importance](Marketing/outputs/04_marketing_ml_diagnostics.png)

### 3.3 Smooth Markov Subscriber Cohort & External Risk Distributions ([`configs/externals.toml`](configs/externals.toml))
- **Paying Subscribers Only**: Free registered accounts (`1,150`) are excluded from revenue. Paying members (`M_t`) evolve smoothly via monthly churn `N(4.8%, 1.0%)` and non-member conversion `N(8.2%, 1.4%)` (conservative vs. `31%` survey intent): **`351` (`M5`) -> `1,546` (`M12`) -> `2,460` (`M24`) -> `3,905` (`M60`)**.
- **External Factors (`externals.toml`)**: Every trajectory compounds monthly **overall pickleball trend shifts** (`N(+0.3%, 1.8%, [-6%, +5%])`), **competitor court drag**, **weather shocks**, **cost inflation**, and **noise complaint legal handling costs**.

![Membership Growth, Year 3 Profit Distribution & External Risk Shocks](FinancialModel/outputs/05_membership_and_monte_carlo_distributions.png)

### 3.4 Permanent `-30%` Demand Stress Test from Day 1
- Even if total demand across all revenue lines is permanently **`30%` lower from Month 1 to Month 60** while fixed leases, payroll, and `7,000M VND` CapEx stay at `100%`:
  1. **Zero Negative Cash Months**: Minimum cash across 60 months is **`+217.59M VND` (`Jun 2027`)**, and stays `>= +1,244.54M VND` through the Q4 2027 Venue 2 construction window.
  2. **Still Clears the Investor Hurdle by `2.94x`**: Stress Year 3 EBITDA reaches **`+10,285.59M VND`** vs. the `3,500M VND` target.

![Monthly Cash Balance Trajectory, -30% Stress Test & External Multipliers](FinancialModel/outputs/06_monthly_cash_balance_and_externals.png)

---

## 4. What Defines Success? (Leveled Cash Allocation, Engineering Critical Paths & Year 3 Targets)

### 4.1 Zero Lump-Sum Cash Burn: Two-Tranche Equity & Leveled Disbursements
- **Tranche 1 (`+500.0M VND` in `Month 1 / Jan 2027`)**: Drawn upon signing (`500M` existing + `500M` draw = `1,000M VND` starting cash) while `PICK! Social` CapEx (`296.95M VND`) is paid in **6 leveled monthly installments of `49.49M VND/mo`** (`Jan–Jun 2027`, `CV = 0.90%`).
- **Tranche 2 (`+7,500.0M VND` in `Month 10 / Oct 2027`)**: Drawn at Q4 expansion kickoff (`7,000M` fit-out paid in **4 equal `-1,750.0M VND/mo` milestones** across `Oct 2027–Jan 2028` + `500M` reserve), after `PICK! Social` has already built cash reserves to **`+2,438.49M VND`**.

![PICK! Social 24-Week Production Gantt Schedule](PickSocialProjectTimeline/outputs/01_pick_social_platform_gantt_chart.png)

![PICK! Social Buffered Cost Estimation & Leveled Monthly Allocation](PickSocialProjectTimeline/outputs/03_cost_estimation_and_leveled_allocation.png)

### 4.2 Venue 2 Construction Timeline & 20-Court Capacity Scaling (`Months 10–13`)
- **16-Week Critical Path (`Oct 2027–Jan 2028`)**: Sequences site handover, `28-day` concrete slab moisture curing (`Weeks 5–8`), 5-layer ITF acrylic surfacing (`Weeks 9–11`), acoustic barriers, and PCCC fire acceptance (`Weeks 13–14`) prior to the `Month 14 (Feb 2028)` opening.

![Venue 2 14-Court Q4 2027 Construction Gantt Schedule](VenueScalingTimeline/outputs/01_venue_scaling_gantt_chart.png)

![20-Court Capacity Scaling & Fill-Up Ramp](VenueScalingTimeline/outputs/02_20court_capacity_and_fill_up_ramp.png)

![Venue 2 Leveled CapEx Allocation & 20-Court Unit Economics](VenueScalingTimeline/outputs/03_capex_allocation_and_unit_economics.png)

### 4.3 Measurable Success Targets Through Year 3

| Gate | Target Date | Operational & Customer KPI | Financial & Cash KPI |
| :--- | :--- | :--- | :--- |
| **Gate 1: Platform Ready** | `Month 4` (`Apr 2027`) | `PICK! Social` passes `500`-user rush load test; Venue 1 utilization **`>= 68.0%`** (`midday >= 45%`). | Platform CapEx `<= 296.95M VND`; ending cash **`>= 1,000.0M VND`** (`1,000.73M` realized). |
| **Gate 2: Pre-Expansion Gate** | `Month 9` (`Sep 2027`) | **`>= 1,350` paying subscribers** (`1,422` expected); monthly churn `<= 5.0%`. | Monthly EBITDA **`>= 400.0M VND/mo`** (`434.7M` expected); cash **`>= 2,200.0M VND`** (`2,438.49M` expected). |
| **Gate 3: 20-Court Ramp** | `Month 24` (`Dec 2028`) | `20`-court utilization reaches **`>= 68.0%`**; paying members exceed **`2,400`** (`2,460` expected). | **Year 2 (`FY 2028`) EBITDA reaches `13,134.12M VND`** (`+5,259.4M` under `-30%` stress). |
| **Gate 4: Investor Target** | `Month 36` (`Dec 2029`) | **`3,167` paying members** (`1,263.6M VND/mo` subscription rev) + **`19,800+` visits/mo**. | **Year 3 (`FY 2029`) EBITDA reaches `20,473.28M VND`** (**`5.85x` hurdle**; `+10,285.6M` under `-30%` stress). |

![Granular Monthly Revenue, COGS, OpEx & SG&A Breakdown (M1 - M60)](FinancialModel/outputs/01_revenue_cogs_opex_sga_sheet.png)

![Monthly Operating Profit & Cash Reserve Schedule (M1 - M60)](FinancialModel/outputs/02_ebitda_and_cash_flow_sheet.png)

---

## 5. Repository Structure & Reproducibility

Run the entire 5-stage simulation, ML training, and chart-generation pipeline from the root directory with a single command:

```bash
uv run main
```

| Directory / File | Description |
| :--- | :--- |
| [`configs/`](configs) | Central stochastic distribution configs (`pricing.toml`, `customers.toml`, `revenue.toml`, `cogs.toml`, `opex.toml`, `capex.toml`, `externals.toml`, `randomness.toml`). |
| [`AdaptivePricing/`](AdaptivePricing) | Stage 1: Hourly court demand elasticity simulation & `HistGradientBoostingRegressor` dynamic pricing engine (`Test R^2 = 0.9922`). |
| [`PickSocialProjectTimeline/`](PickSocialProjectTimeline) | Stage 2: 24-week software engineering WBS, Gantt schedule, buffered cost estimation (`296.95M VND`), and leveled monthly allocation. |
| [`VenueScalingTimeline/`](VenueScalingTimeline) | Stage 3: Q4 2027 14-court expansion WBS, Gantt schedule, leveled `-1,750M VND/mo` CapEx schedule, and 20-court ramp model. |
| [`Marketing/`](Marketing) | Stage 4: Follower-to-customer funnel (`8.57%`), 4-segment `MultiOutputRegressor` budget predictor (`Test R^2 = 0.9555`), and 12 ancillary spending distributions. |
| [`FinancialModel/`](FinancialModel) | Stage 5: 60-month (`Jan 2027 – Dec 2031`), `N = 300` Monte Carlo 3-statement financial model, stochastic baseline comparison, and `-30%` demand stress test. |
