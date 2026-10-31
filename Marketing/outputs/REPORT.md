# PICK! Marketing Funnel, Segment VND/Visit Response & 12 Ancillary Revenue Distributions Report

## 1. Explicit Team Assumption: Social Follower-to-Customer Funnel (~8.57% Conversion)

Per **Case Page 2 ("PICK! At a glance")** and **Case Exhibit 2 ("Monthly Costs")**, when PICK! spends its baseline **VND 25,000,000 / month** on marketing, it sustains:
- **42,000** TikTok and Instagram followers
- **3,600** monthly unique customers
- **7,596** exact monthly player visits (reported as *"about 7,600 player visits"* in Case Page 2)

> **EXPLICIT TEAM ASSUMPTION (NOTED IN `configs/customers.toml`)**:
> We explicitly assume that PICK!'s active monthly unique customers are strictly a **subset of its TikTok and Instagram followers**. Therefore, the baseline **Follower-to-Customer Conversion Rate** is parameterized as a Truncated Normal distribution centered at:
> $$\eta_0 = \frac{3,600\text{ unique customers}}{42,000\text{ followers}} = 8.5714\% \quad (\text{Mean} = 8.5714\%,\; \text{Std} = 0.38\%,\; \text{Range} = [7.40\%,\; 9.80\%])$$

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

---

## 4. 60-Month ML Marketing Allocation & Ancillary Revenue Trajectory Summary

| Horizon Milestone | Active Courts | ML Marketing Spend (Comp / Soc / Trend / Fam = **Total**) | Active Followers (`~8.57%` Conv) | Total Monthly Visits | Paddle & Balls Rev (`M VND`) | Coaching Rev (`M VND`) | Food & Drinks Rev (`M VND`) | Total Ancillary Rev (`M VND`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Month 1 (Jan-27, Build)** | 6 | 10.27M / 16.55M / 13.29M / 8.36M = **48.47M** | 42,871 | 9,196 | 18.54M | 137.59M | 91.66M | **247.79M** |
| **Month 5 (May-27, App Live)** | 6 | 8.54M / 13.91M / 13.06M / 6.75M = **42.26M** | 46,307 | 9,717 | 17.20M | 148.69M | 97.07M | **262.95M** |
| **Month 14 (Feb-28, 20 Courts)** | 20 | 18.48M / 23.93M / 23.88M / 15.97M = **82.26M** | 54,999 | 20,715 | 40.31M | 307.26M | 209.31M | **556.88M** |
| **Month 24 (Dec-28, End Y2)** | 20 | 10.16M / 15.57M / 17.14M / 7.68M = **50.55M** | 68,733 | 20,827 | 41.58M | 324.91M | 201.80M | **568.28M** |
| **Month 36 (Dec-29, End Y3)** | 20 | 10.39M / 16.40M / 17.34M / 7.98M = **52.12M** | 87,165 | 21,199 | 40.79M | 302.90M | 201.78M | **545.47M** |
| **5-Year Monthly Average** | 17.7 | 10.86M / 16.87M / 17.46M / 8.24M = **53.43M** | 83,037 | 18,795 | 36.75M | 284.51M | 185.09M | **506.35M** |
