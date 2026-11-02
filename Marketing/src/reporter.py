"""Export CSV schedules, `REPORT.md`, and `ML_METRICS.md` for `Marketing/` (consumed by `FinancialModel/`)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import MarketingAppConfig
from .model import MarketingMLEvaluationResults
from .simulator import MarketingSimulationResults


def export_marketing_outputs(
    cfg: MarketingAppConfig,
    sim: MarketingSimulationResults,
    ml_res: MarketingMLEvaluationResults,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """
    Export the 4 required downstream artifacts in `Marketing/outputs/`:
      1. `monthly_segment_marketing_and_visits_60m.csv`
      2. `segment_ancillary_revenue_60m.csv`
      3. `REPORT.md`
      4. `ML_METRICS.md`
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    T = sim.n_months
    months_1based = np.arange(1, T + 1)

    ml_b_seg = np.mean(ml_res.ml_mkt_budgets_seg, axis=0)      # (60, 4)
    ml_b_tot = np.mean(ml_res.ml_mkt_budget_total, axis=0)     # (60,)
    ml_b_p05 = np.percentile(ml_res.ml_mkt_budget_total, 5, axis=0)
    ml_b_p95 = np.percentile(ml_res.ml_mkt_budget_total, 95, axis=0)
    base_b_seg = np.mean(sim.base_mkt_budgets_seg, axis=0)     # (60, 4)
    base_b_tot = np.sum(base_b_seg, axis=1)

    ml_fol = np.mean(ml_res.ml_followers, axis=0)
    base_fol = np.mean(sim.base_followers, axis=0)
    conv_rate = np.mean(sim.sampled_conv_rate, axis=0)

    ml_cust_seg = np.mean(ml_res.ml_customers_seg, axis=0)     # (60, 4)
    ml_cust_tot = np.mean(ml_res.ml_customers_total, axis=0)
    base_cust_tot = np.mean(sim.base_customers_total, axis=0)

    ml_vis_seg = np.mean(ml_res.ml_visits_seg, axis=0)         # (60, 4)
    ml_vis_tot = np.mean(ml_res.ml_visits_total, axis=0)
    ml_vis_p05 = np.percentile(ml_res.ml_visits_total, 5, axis=0)
    ml_vis_p95 = np.percentile(ml_res.ml_visits_total, 95, axis=0)
    base_vis_seg = np.mean(sim.base_visits_seg, axis=0)        # (60, 4)
    base_vis_tot = np.mean(sim.base_visits_total, axis=0)

    df_mkt = pd.DataFrame(
        {
            "Month_Index": months_1based,
            "Month_Label": list(sim.month_labels),
            "Active_Courts": np.mean(sim.sampled_active_courts, axis=0).astype(int),
            "Social_Followers_ML": np.round(ml_fol, 1),
            "Social_Followers_Baseline": np.round(base_fol, 1),
            "Follower_to_Customer_Conv_Rate": np.round(conv_rate, 6),
            "Unique_Customers_Competitive_ML": np.round(ml_cust_seg[:, 0], 1),
            "Unique_Customers_Social_ML": np.round(ml_cust_seg[:, 1], 1),
            "Unique_Customers_Trend_Explorers_ML": np.round(ml_cust_seg[:, 2], 1),
            "Unique_Customers_Families_ML": np.round(ml_cust_seg[:, 3], 1),
            "Unique_Customers_Total_ML": np.round(ml_cust_tot, 1),
            "Unique_Customers_Total_Baseline": np.round(base_cust_tot, 1),
            "Visits_Competitive_ML": np.round(ml_vis_seg[:, 0], 1),
            "Visits_Social_ML": np.round(ml_vis_seg[:, 1], 1),
            "Visits_Trend_Explorers_ML": np.round(ml_vis_seg[:, 2], 1),
            "Visits_Families_ML": np.round(ml_vis_seg[:, 3], 1),
            "Visits_Total_ML": np.round(ml_vis_tot, 1),
            "Visits_Total_P05_ML": np.round(ml_vis_p05, 1),
            "Visits_Total_P95_ML": np.round(ml_vis_p95, 1),
            "Visits_Competitive_Baseline": np.round(base_vis_seg[:, 0], 1),
            "Visits_Social_Baseline": np.round(base_vis_seg[:, 1], 1),
            "Visits_Trend_Explorers_Baseline": np.round(base_vis_seg[:, 2], 1),
            "Visits_Families_Baseline": np.round(base_vis_seg[:, 3], 1),
            "Visits_Total_Baseline": np.round(base_vis_tot, 1),
            "Mkt_Budget_Competitive_ML_MVND": np.round(ml_b_seg[:, 0], 4),
            "Mkt_Budget_Social_ML_MVND": np.round(ml_b_seg[:, 1], 4),
            "Mkt_Budget_Trend_Explorers_ML_MVND": np.round(ml_b_seg[:, 2], 4),
            "Mkt_Budget_Families_ML_MVND": np.round(ml_b_seg[:, 3], 4),
            "Mkt_Budget_Total_ML_MVND": np.round(ml_b_tot, 4),
            "Mkt_Budget_Total_P05_ML_MVND": np.round(ml_b_p05, 4),
            "Mkt_Budget_Total_P95_ML_MVND": np.round(ml_b_p95, 4),
            "Mkt_Budget_Competitive_Base_MVND": np.round(base_b_seg[:, 0], 4),
            "Mkt_Budget_Social_Base_MVND": np.round(base_b_seg[:, 1], 4),
            "Mkt_Budget_Trend_Explorers_Base_MVND": np.round(base_b_seg[:, 2], 4),
            "Mkt_Budget_Families_Base_MVND": np.round(base_b_seg[:, 3], 4),
            "Mkt_Budget_Total_Base_MVND": np.round(base_b_tot, 4),
        }
    )
    mkt_csv_path = output_dir / "monthly_segment_marketing_and_visits_60m.csv"
    df_mkt.to_csv(mkt_csv_path, index=False)

    # 2. Export `segment_ancillary_revenue_60m.csv` (12 Segment-by-Ancillary lines + 3 Totals)
    pad_seg = np.mean(ml_res.ml_paddle_rev_seg, axis=0)       # (60, 4)
    coa_seg = np.mean(ml_res.ml_coaching_rev_seg, axis=0)     # (60, 4)
    fnb_seg = np.mean(ml_res.ml_fnb_rev_seg, axis=0)          # (60, 4)

    pad_tot = np.mean(ml_res.ml_paddle_rev_total, axis=0)
    coa_tot = np.mean(ml_res.ml_coaching_rev_total, axis=0)
    fnb_tot = np.mean(ml_res.ml_fnb_rev_total, axis=0)
    anc_tot = np.mean(ml_res.ml_ancillary_rev_total, axis=0)

    base_pad_tot = np.mean(np.sum(sim.base_paddle_rev_seg, axis=2), axis=0)
    base_coa_tot = np.mean(np.sum(sim.base_coaching_rev_seg, axis=2), axis=0)
    base_fnb_tot = np.mean(np.sum(sim.base_fnb_rev_seg, axis=2), axis=0)
    base_anc_tot = np.mean(sim.base_ancillary_rev_total, axis=0)

    df_anc = pd.DataFrame(
        {
            "Month_Index": months_1based,
            "Month_Label": list(sim.month_labels),
            "Paddle_Balls_Competitive_MVND": np.round(pad_seg[:, 0], 4),
            "Paddle_Balls_Social_MVND": np.round(pad_seg[:, 1], 4),
            "Paddle_Balls_Trend_Explorers_MVND": np.round(pad_seg[:, 2], 4),
            "Paddle_Balls_Families_MVND": np.round(pad_seg[:, 3], 4),
            "Paddle_Balls_Total_ML_MVND": np.round(pad_tot, 4),
            "Paddle_Balls_Total_Base_MVND": np.round(base_pad_tot, 4),
            "Coaching_Competitive_MVND": np.round(coa_seg[:, 0], 4),
            "Coaching_Social_MVND": np.round(coa_seg[:, 1], 4),
            "Coaching_Trend_Explorers_MVND": np.round(coa_seg[:, 2], 4),
            "Coaching_Families_MVND": np.round(coa_seg[:, 3], 4),
            "Coaching_Total_ML_MVND": np.round(coa_tot, 4),
            "Coaching_Total_Base_MVND": np.round(base_coa_tot, 4),
            "Food_Drinks_Competitive_MVND": np.round(fnb_seg[:, 0], 4),
            "Food_Drinks_Social_MVND": np.round(fnb_seg[:, 1], 4),
            "Food_Drinks_Trend_Explorers_MVND": np.round(fnb_seg[:, 2], 4),
            "Food_Drinks_Families_MVND": np.round(fnb_seg[:, 3], 4),
            "Food_Drinks_Total_ML_MVND": np.round(fnb_tot, 4),
            "Food_Drinks_Total_Base_MVND": np.round(base_fnb_tot, 4),
            "Ancillary_Total_ML_MVND": np.round(anc_tot, 4),
            "Ancillary_Total_Base_MVND": np.round(base_anc_tot, 4),
        }
    )
    anc_csv_path = output_dir / "segment_ancillary_revenue_60m.csv"
    df_anc.to_csv(anc_csv_path, index=False)

    # 3. Export `REPORT.md`
    m = ml_res.ml_metrics
    report_md = f"""# PICK! Marketing Funnel, Segment VND/Visit Response & 12 Ancillary Revenue Distributions Report

## 1. Explicit Team Assumption: Social Follower-to-Customer Funnel (~8.57% Conversion)

Per **Case Page 2 ("PICK! At a glance")** and **Case Exhibit 2 ("Monthly Costs")**, when PICK! spends its baseline **VND 25,000,000 / month** on marketing, it sustains:
- **42,000** TikTok and Instagram followers
- **3,600** monthly unique customers
- **7,596** exact monthly player visits (reported as *"about 7,600 player visits"* in Case Page 2)

> **EXPLICIT TEAM ASSUMPTION (NOTED IN `configs/customers.toml`)**:
> We explicitly assume that PICK!'s active monthly unique customers are strictly a **subset of its TikTok and Instagram followers**. Therefore, the baseline **Follower-to-Customer Conversion Rate** is parameterized as a Truncated Normal distribution centered at:
> $$\\eta_0 = \\frac{{3,600\\text{{ unique customers}}}}{{42,000\\text{{ followers}}}} = 8.5714\\% \\quad (\\text{{Mean}} = 8.5714\\%,\\; \\text{{Std}} = 0.38\\%,\\; \\text{{Range}} = [7.40\\%,\\; 9.80\\%])$$

---

## 2. Segment Marketing Cost per Visit (`x, y, z, w` VND/Visit) & Baseline 25.00M VND Split

Splitting the baseline **25.00M VND/month** marketing budget across the 4 customer segments from **Case Exhibit 4** and merging customer count and visits/month into direct **Monthly Segment Visits Distributions** yields the exact marketing spend required to drive **1 player visit** in each segment:

| Customer Segment | Share of Customers | Baseline Customers | Visits / Customer / Mo | Merged Monthly Visits Dist (Mean +/- SD) | Baseline Marketing Budget Split | Marketing Cost per Visit (`VND / Visit`) | Marketing Cost per Active Customer (`VND / Cust`) | Strategic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Competitive Players** | 20.0% | 720 | 4.5 | **3,240 +/- 165** | **5.00M VND** (20.0%) | **1,543 VND / visit** | 6,944 VND / cust | Lowest cost/visit due to high 4.5x/mo repeat frequency |
| **2. Social Players** | 35.0% | 1,260 | 2.0 | **2,520 +/- 135** | **8.50M VND** (34.0%) | **3,373 VND / visit** | 6,746 VND / cust | Core group-play & PICK! Social membership audience |
| **3. Trend Explorers** | 30.0% | 1,080 | 1.1 | **1,188 +/- 95** | **7.50M VND** (30.0%) | **6,313 VND / visit** | 6,944 VND / cust | Highest cost/visit (4.09x Competitive) due to 1.1x/mo low retention |
| **4. Families** | 15.0% | 540 | 1.2 | **648 +/- 48** | **4.00M VND** (16.0%) | **6,173 VND / visit** | 7,407 VND / cust | High acquisition cost per visit, but highest ancillary coaching & F&B spend |
| **Total / Weighted Avg** | **100.0%** | **3,600** | **2.11** | **7,596 (~7,600)** | **25.00M VND (100%)** | **3,291 VND / visit** | **6,944 VND / cust** | Exact reconciliation to Case Page 2, Exhibit 2 & Exhibit 4 |

---

## 3. Grounding the 12 Segment-by-Ancillary Spending Distributions (`3 Lines x 4 Segments`)

To model **Revenue Line 2 (Paddle Rental & Balls = 15.00M VND/mo)**, **Revenue Line 3 (Coaching = 115.00M VND/mo)**, and **Revenue Line 4 (Food & Drinks = 75.00M VND/mo)** dynamically from customer segment visits, we define **12 Truncated Normal distributions** in `configs/customers.toml` (`3 ancillary lines x 4 customer segments`), grounded simultaneously in **Case Exhibit 1** and **Case Exhibit 4**:

| Customer Segment | Baseline Visits / Mo | Dist 1: Paddle & Balls (`kVND/visit` -> `M VND/mo`) | Dist 2: Coaching (`kVND/visit` -> `M VND/mo`) | Dist 3: Food & Drinks (`kVND/visit` -> `M VND/mo`) | Total Ancillary (`kVND/visit`) | Court Hire Share (`kVND/visit`) | Total Spend / Visit (**Case Exhibit 4**) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Competitive Players** | 3,240 | **0.80 +/- 0.12 k** (2.59M/mo) | **21.30 +/- 1.85 k** (69.01M/mo) | **10.00 +/- 0.95 k** (32.40M/mo) | 32.10 k | 62.90 k | **95.00 kVND** (Exh. 4 = 95k) |
| **2. Social Players** | 2,520 | **1.60 +/- 0.20 k** (4.03M/mo) | **9.52 +/- 0.90 k** (23.99M/mo) | **11.50 +/- 1.05 k** (28.98M/mo) | 22.62 k | 57.38 k | **80.00 kVND** (Exh. 4 = 80k) |
| **3. Trend Explorers** | 1,188 | **5.20 +/- 0.50 k** (6.18M/mo) | **4.80 +/- 0.55 k** (5.70M/mo) | **5.99 +/- 0.65 k** (7.12M/mo) | 15.99 k | 49.01 k | **65.00 kVND** (Exh. 4 = 65k) |
| **4. Families** | 648 | **3.40 +/- 0.35 k** (2.20M/mo) | **25.15 +/- 2.10 k** (16.30M/mo) | **10.03 +/- 0.95 k** (6.50M/mo) | 38.58 k | 46.42 k | **85.00 kVND** (Exh. 4 = 85k) |
| **Baseline Monthly Sum** | **7,596** | **15.005M VND/mo** (Exh. 1 = **15.00M**) | **115.001M VND/mo** (Exh. 1 = **115.00M**) | **74.995M VND/mo** (Exh. 1 = **75.00M**) | **205.00M/mo** | **436.70M/mo** | **641.70M VND/mo** (Exh. 1 = **640M**) |

### 3.1 Quarterly Seasonal Tournament Ancillary & Follower Boost (`Mar, Jun, Sep, Dec` from Month 6+)
Starting in **Month 6 (`Jun-27`, Summer Tournament)** after `PICK! Social` launches in `Month 5 (May-27)`, PICK! hosts a flagship seasonal tournament every quarter (`Spring [Mar]`, `Summer [Jun]`, `Fall [Sep]`, `Winter [Dec]`):
- **Tournament Participant Field**: `192 +/- 16` players in the `6`-court phase (`Months 6, 9, 12`), scaling to `420 +/- 32` players across the `20`-court network (`Months 15, 18, ..., 60`).
- **Social Follower Boost**: Tournament highlight reels and brackets multiply monthly new follower acquisition by **`1.15x` (`+15%`)** in tournament months.
- **Ancillary Spend Lift (`+85.0k VND / participant`)**: Both active members (who enter for **free**) and non-members (who pay **`350,000 VND`** entry fee) generate incremental full-day tournament ancillary spending of **`55.0k VND`** in `Food & Drinks`, **`10.0k VND`** in `Paddle & Balls`, and **`20.0k VND`** in pre-tournament `Coaching` clinics (`~16.3M VND/qtr` in 6-court phase; `~35.7M VND/qtr` in 20-court phase).

---

## 4. 60-Month ML Marketing Allocation & Ancillary Revenue Trajectory Summary

| Horizon Milestone | Active Courts | ML Marketing Spend (Comp / Soc / Trend / Fam = **Total**) | Active Followers (`~8.57%` Conv) | Total Monthly Visits | Paddle & Balls Rev (`M VND`) | Coaching Rev (`M VND`) | Food & Drinks Rev (`M VND`) | Total Ancillary Rev (`M VND`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Month 1 (Jan-27, Build)** | 6 | {ml_b_seg[0,0]:.2f}M / {ml_b_seg[0,1]:.2f}M / {ml_b_seg[0,2]:.2f}M / {ml_b_seg[0,3]:.2f}M = **{ml_b_tot[0]:.2f}M** | {ml_fol[0]:,.0f} | {ml_vis_tot[0]:,.0f} | {pad_tot[0]:.2f}M | {coa_tot[0]:.2f}M | {fnb_tot[0]:.2f}M | **{anc_tot[0]:.2f}M** |
| **Month 5 (May-27, App Live)** | 6 | {ml_b_seg[4,0]:.2f}M / {ml_b_seg[4,1]:.2f}M / {ml_b_seg[4,2]:.2f}M / {ml_b_seg[4,3]:.2f}M = **{ml_b_tot[4]:.2f}M** | {ml_fol[4]:,.0f} | {ml_vis_tot[4]:,.0f} | {pad_tot[4]:.2f}M | {coa_tot[4]:.2f}M | {fnb_tot[4]:.2f}M | **{anc_tot[4]:.2f}M** |
| **Month 14 (Feb-28, 20 Courts)** | 20 | {ml_b_seg[13,0]:.2f}M / {ml_b_seg[13,1]:.2f}M / {ml_b_seg[13,2]:.2f}M / {ml_b_seg[13,3]:.2f}M = **{ml_b_tot[13]:.2f}M** | {ml_fol[13]:,.0f} | {ml_vis_tot[13]:,.0f} | {pad_tot[13]:.2f}M | {coa_tot[13]:.2f}M | {fnb_tot[13]:.2f}M | **{anc_tot[13]:.2f}M** |
| **Month 24 (Dec-28, End Y2)** | 20 | {ml_b_seg[23,0]:.2f}M / {ml_b_seg[23,1]:.2f}M / {ml_b_seg[23,2]:.2f}M / {ml_b_seg[23,3]:.2f}M = **{ml_b_tot[23]:.2f}M** | {ml_fol[23]:,.0f} | {ml_vis_tot[23]:,.0f} | {pad_tot[23]:.2f}M | {coa_tot[23]:.2f}M | {fnb_tot[23]:.2f}M | **{anc_tot[23]:.2f}M** |
| **Month 36 (Dec-29, End Y3)** | 20 | {ml_b_seg[35,0]:.2f}M / {ml_b_seg[35,1]:.2f}M / {ml_b_seg[35,2]:.2f}M / {ml_b_seg[35,3]:.2f}M = **{ml_b_tot[35]:.2f}M** | {ml_fol[35]:,.0f} | {ml_vis_tot[35]:,.0f} | {pad_tot[35]:.2f}M | {coa_tot[35]:.2f}M | {fnb_tot[35]:.2f}M | **{anc_tot[35]:.2f}M** |
| **5-Year Monthly Average** | 17.7 | {np.mean(ml_b_seg[:,0]):.2f}M / {np.mean(ml_b_seg[:,1]):.2f}M / {np.mean(ml_b_seg[:,2]):.2f}M / {np.mean(ml_b_seg[:,3]):.2f}M = **{np.mean(ml_b_tot):.2f}M** | {np.mean(ml_fol):,.0f} | {np.mean(ml_vis_tot):,.0f} | {np.mean(pad_tot):.2f}M | {np.mean(coa_tot):.2f}M | {np.mean(fnb_tot):.2f}M | **{np.mean(anc_tot):.2f}M** |
"""
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")

    # 4. Export `ML_METRICS.md`
    ml_md = f"""# Segment Marketing Budget ML Predictor - Technical Evaluation (`ML_METRICS.md`)

## 1. Model Architecture & Train/Test Split

- **Estimator**: 4 Segment-Specific `HistGradientBoostingRegressor` models (`max_iter=220`, `learning_rate=0.065`, `max_depth=6`, `l2_regularization=1.2`)
- **Training Split (80%)**: `{sim.n_train}` Monte Carlo trajectories x `60` months = **`{m.n_train_obs:,}` monthly state transitions**
- **Held-Out Test Split (20%)**: `{sim.n_test}` Monte Carlo trajectories x `60` months = **`{m.n_test_obs:,}` monthly state transitions**
- **Input Features (12)**: Last month's active unique customers by segment (4), last month's player visits by segment (4), social media followers (`k`), follower-to-customer conversion rate (`~8.57%`), cumulative pickleball trend index ($I_t$), and active court capacity & Venue 2 fill headroom (`6` vs `20` courts).

## 2. Overall & Per-Segment Out-of-Sample Regression Metrics

| Evaluation Split / Segment Target | $R^2$ Score | RMSE (`M VND`) | MAE (`M VND`) | MAPE (`%`) |
| :--- | :---: | :---: | :---: | :---: |
| **In-Sample Training Set (`N = {m.n_train_obs:,}`)** | **{m.train_r2:.4f}** | {m.train_rmse_mvnd:.4f} | {m.train_mae_mvnd:.4f} | {m.train_mape_pct:.2f}% |
| **Out-of-Sample Test Set - Overall (`N = {m.n_test_obs:,}`)** | **{m.test_r2:.4f}** | **{m.test_rmse_mvnd:.4f}** | **{m.test_mae_mvnd:.4f}** | **{m.test_mape_pct:.2f}%** |
| Out-of-Sample: **Competitive Players Budget** | {m.seg_test_r2['competitive']:.4f} | {m.seg_test_rmse_mvnd['competitive']:.4f} | {m.seg_test_mae_mvnd['competitive']:.4f} | {m.seg_test_mape_pct['competitive']:.2f}% |
| Out-of-Sample: **Social Players Budget** | {m.seg_test_r2['social']:.4f} | {m.seg_test_rmse_mvnd['social']:.4f} | {m.seg_test_mae_mvnd['social']:.4f} | {m.seg_test_mape_pct['social']:.2f}% |
| Out-of-Sample: **Trend Explorers Budget** | {m.seg_test_r2['trend_explorers']:.4f} | {m.seg_test_rmse_mvnd['trend_explorers']:.4f} | {m.seg_test_mae_mvnd['trend_explorers']:.4f} | {m.seg_test_mape_pct['trend_explorers']:.2f}% |
| Out-of-Sample: **Families Budget** | {m.seg_test_r2['families']:.4f} | {m.seg_test_rmse_mvnd['families']:.4f} | {m.seg_test_mae_mvnd['families']:.4f} | {m.seg_test_mape_pct['families']:.2f}% |

- **Oracle Net Contribution Capture on Held-Out Test Set**: **`{m.oracle_net_contrib_capture_pct:.2f}%`**
- **Net Contribution Uplift Capture vs. Static 25M Baseline**: **`{m.uplift_capture_vs_oracle_pct:.2f}%`**
"""
    ml_metrics_path = output_dir / "ML_METRICS.md"
    ml_metrics_path.write_text(ml_md, encoding="utf-8")

    return mkt_csv_path, anc_csv_path, report_path, ml_metrics_path
