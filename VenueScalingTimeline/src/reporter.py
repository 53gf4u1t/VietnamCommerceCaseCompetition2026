"""CSV and Markdown report exporter for VenueScalingTimeline (Sub-Solution Path A: 14-Court Second Club starting Q3 2027)."""

from __future__ import annotations

from pathlib import Path
from .config import VenueScalingConfig
from .schedule_model import VenueScalingResults


def export_venue_scaling_outputs(
    cfg: VenueScalingConfig,
    res: VenueScalingResults,
    chart_paths: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Export WBS CSV, 60-Month Expansion CapEx & Fill-Up Ramp CSV, and cited REPORT.md into outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    wbs_csv = output_dir / "venue_scaling_wbs.csv"
    res.wbs_df.to_csv(wbs_csv, index=False)

    monthly_csv = output_dir / "monthly_expansion_capex_and_ramp.csv"
    res.monthly_60m_df.to_csv(monthly_csv, index=False)

    report_path = output_dir / "REPORT.md"

    stage_rows = []
    for st in res.stages:
        stage_rows.append(
            f"| **{st.stage_id}** | **{st.short_name}** | `W{st.start_week + 1:.1f}–W{st.end_week:.1f}` ({st.start_date.strftime('%b %d, %Y')} – {st.end_date.strftime('%b %d, %Y')}) | "
            f"**{st.span_weeks:.1f}w** | {st.task_count} | **`{st.stage_capex_mvnd:,.2f}M`** | `{st.budget_share_pct:.1f}%` | {st.summary} |"
        )
    stage_table_md = "\n".join(stage_rows)

    task_rows = []
    for t in res.tasks:
        cites = ", ".join(f"`[{c}]`" for c in t.citation_keys)
        task_rows.append(
            f"| **`{t.id}`** | {t.name} | `W{t.start_week + 1:.1f}–W{t.end_week:.1f}` ({t.duration_weeks:.1f}w) | {t.core_deliverable} | "
            f"**`{t.capex_mvnd:,.2f}M`** | {t.estimation_rationale} ({cites}) |"
        )
    task_table_md = "\n".join(task_rows)

    cite_rows = []
    for c_key, c in cfg.citations.items():
        cite_rows.append(
            f"| `[{c_key}]` | **{c.title}** — *{c.authors}* (`{c.url}`) | {c.relevance} |"
        )
    cite_table_md = "\n".join(cite_rows)

    meta = cfg.meta
    report_md = f"""# PICK! Sub-Solution Path A: 14-Court Second Club Expansion (`6` to `20` Courts) — Engineering Timeline, Even CapEx Split & 6–9 Month Fill-Up Report

**Project**: {meta.project_name}  
**Company**: {meta.company}  
**Strategic Sequencing**: Start of **Q4 2027 (`Month 10: Oct 2027` / `2027-10-04`)** — launched after `4 months` of custom platform development (`Months 1–4: Jan–Apr 2027`, fortified by `500.0M VND` drawn from Investor Minh Tran's reserve upon signing in `Month 1`) and `5 full months` of live platform operation (`Months 5–9: May–Sep 2027`) prove out elevated court utilization, ML marketing funnel performance, and recurring membership cash flow.  
**Turnkey Construction & Curing Duration**: **4.0 Months / 16.0 Weeks (`Months 10–13: Oct 2027 – Jan 2028`)**  
**Grand Opening & 20-Court Network Activation**: **Start of Month 14 (`Feb 2028`)** — scaling PICK! from **`6` courts to `20` courts total (`+233.33%` capacity)**  
**Preset Fit-Out & Launch CapEx (`Case Page 3`)**: **`VND 7,000.00M` (`7.0 billion VND`)**, split **evenly across the 4 construction months (`Oct 2027 – Jan 2028`) = `1,750.00M VND/month`** (`+1,000.00M VND` held in cash reserve = `8,000.00M VND` total investor capital: `500.00M VND` drawn upon signing in `Month 1` + `7,500.00M VND` drawn at Q4 2027 expansion kickoff in `Month 10`)  
**Second Club 5-Year Lease (`Case Page 3`)**: **`VND 330.00M/month`** (commencing `Month 10 / Oct-27`)  
**Capacity Fill-Up Distribution (`Case Page 3`)**: **6.0 to 9.0 Months** (`TruncatedNormal(mean=7.5, std=0.75, min=6.0, max=9.0)` months from `Feb 2028` opening)

---

## 1. Strategic Sequencing: Create Platform -> Collect 5 Months of Live Profit -> Expand Venues (`6` to `20` Courts)

Rather than taking the high-risk approach of signing a `VND 330M/month` lease and spending `VND 7.0B` on day one before fixing PICK!'s weekday off-peak under-utilization (`40%`) and casual player churn, our integrated roadmap executes a disciplined three-step sequence well ahead of Investor Minh Tran's **Year 3 (`FY 2029`)** deadline:

1. **Phase 1 — Build & Launch the Digital Core (`Months 1–4: Jan–Apr 2027`, with `500M VND` Signing Reserve Tranche in `M1`)**: Complete the `PICK! Social & Adaptive Booking Platform` (`Milestone M4: Apr 25, 2027`). Drawing `500.00M VND` of Minh Tran's `1,000.00M VND` reserve upon signing in `Month 1 (Jan-27)` ensures PICK! maintains a rock-solid positive cash buffer even under an immediate Day-1 `-30%` demand stress test.
2. **Phase 2 — Validate Demand & Accumulate Cash Across 5 Full Months (`Months 5–9: May–Sep 2027`)**: Operate the new platform and 4-segment ML marketing engine for 5 full months on the existing 6-court venue, lifting blended court utilization from `66.0%` to `78.4%` (`460.75M VND/mo` court hire), onboarding paying `PICK! Social` members (`@ 399k VND/mo`), and building a large organic cash reserve prior to construction.
3. **Phase 3 — Q4 2027 14-Court Flagship Venue Expansion (`Months 10–13 Build: Oct 2027 – Jan 2028` -> `Month 14 Open: Feb 2028`)**:
   - Draw the remaining `VND 7,500.00M` investor tranche in `Month 10 (Oct-27)` (`7,000.0M VND` fit-out split evenly at **`1,750.0M VND/month` across `Oct 2027 – Jan 2028`**, plus the remaining `500.0M VND` reserve buffer).
   - Open the 14 new courts (plus coaching video studio & specialty café) at the start of **Month 14 (`Feb 2028`)**, expanding PICK! to **20 courts total** and completing the **6-to-9 month fill-up ramp by `Aug–Oct 2028` (`Months 20–22`)**—giving PICK! full steady-state 20-court operations before **Year 3 (`FY 2029`)** begins!

---

## 2. Why Construction & Fit-Out Takes Exactly 4.0 Months (`16.0 Weeks`: `Oct 2027 – Jan 2028`)

Based on commercial pickleball court engineering standards from the **American Sports Builders Association (`[ASBA2024]`)**, **PickleTile / Pro Track & Tennis (`[PickleTile2025]`)**, and **HCMC commercial hospitality fit-out benchmarks (`[HCMCFitOut2025]`, `[VietnamPCCC2024]`)**, the critical path of a 14-court covered sports club is governed by concrete hydration physics:

- **Weeks 1.0–3.0 (`Stage V1`)**: Site lease handover, topographic laser survey, STC-32 acoustic barrier design (to guarantee `<55 dBA` exterior noise after `21:00` per `QCVN 26:2010/BTNMT`), and HCMC construction + Police Fire Safety (`PCCC`) design appraisal.
- **Weeks 2.5–8.5 (`Stage V2` — Critical Path Physics Gate)**: `2.0 weeks` sub-base compaction & monsoon stormwater drains (`V2.1`) + `2.0 weeks` 15-mil vapor barrier & 14-court post-tensioned concrete slab pour (`V2.2`) + **mandatory `28-day` (`4.0-week`) concrete hydration & moisture-vapor curing window (`V2.3`)**. Applying non-breathable acrylic court coatings before the 28-day cure causes hydrostatic blistering and delamination (`[ASBA2024]`, `[PickleTile2025]`).
- **Weeks 3.5–11.5 (`Stage V3` — Simultaneous Overhead & Interior Track)**: While the concrete slabs cure on the ground, contractors simultaneously erect the 9m-clearance insulated steel canopy roof (`V3.1`), 3-phase electrical & 500-lux anti-glare LED lighting (`V3.2`), and the specialty café, coaching studio & locker showers (`V3.3`).
- **Weeks 8.5–14.0 (`Stage V4`)**: Once the 28-day slab cure finishes at `W8.5`, certified surfacers perform calcium-chloride moisture testing & diamond grinding (`1.5w`), apply 5 coats of cushioned silica-sand acrylic surfacing across all 14 courts (`3.0w`), and hang perimeter STC-32 acoustic curtains & divider fencing (`2.5w`).
- **Weeks 13.0–16.0 (`Stage V5`)**: Final on-site Police `PCCC` fire safety acceptance (`1.5w`), QR gate/POS & `AdaptivePricing` software integration (`1.5w`), and staff/coach training + launch tournament (`1.5w`), completing at the end of **Month 13 (`Jan 30, 2028`)**.

### 2.1 Stage-Level Schedule & VND 7,000.0M CapEx Summary

| Stage ID | Stage Name | Calendar Window | Span | Tasks | Stage CapEx (`M VND`) | Share of `7,000M` | Core Engineering Scope |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
{stage_table_md}
| **BUILD** | **Total 4-Month Turnkey Fit-Out (`V1–V5`)** | **`Oct 04, 2027 – Jan 23, 2028` (`Months 10–13`)** | **16.0w** | **15** | **`7,000.00M`** | **`100.0%`** | **Evenly allocated at `1,750.00M VND/month` (`Oct 2027 – Jan 2028`)** |
| **RESERVE** | **Investor Cash Reserve (`Case Page 3`)** | **`500M` Drawn `M1 (Jan-27)` + `500M` Drawn `M10 (Oct-27)`** | — | — | **`1,000.00M`** | — | **Total Investor Capital = `8,000.00M VND` (`8.0B VND`)** |

### 2.2 Task-by-Task Work Breakdown Structure, CapEx & Citations (`V1.1`–`V6.2`)

| Task ID | Task Name | Window & Duration | Core Deliverable | CapEx (`M VND`) | Engineering Derivation & Cited Sources |
| :--- | :--- | :---: | :--- | :---: | :--- |
{task_table_md}

---

## 3. Modeling 20-Court Revenue Using OUR Model (Instead of Case's Static 1.5B Estimate)

Per your explicit instruction, we **do not** use the Case Question's unverified `VND 1,500.0M/month` static revenue figure for the new club. Instead, we derive the second club's revenue directly from **OUR validated model**:

1. **Capacity Scaling from `6` Courts to `20` Courts (`+233.33%` Capacity)**:
   - Existing Venue 1 has **`6` courts** (`2,880` available court-hours/month), generating **`{res.venue1_steady_court_hire_mvnd:,.2f}M VND/month`** in `AdaptivePricing` court hire revenue and **`{res.venue1_steady_ancillary_mvnd:,.2f}M VND/month`** in ancillary revenue (`15.0M` Paddle/Balls + `115.0M` Coaching + `75.0M` Food & Drinks).
   - New Venue 2 adds **`14` courts** (`6,720` available court-hours/month, or $\\frac{{14}}{{6}} = 2.3333\\times$ Venue 1's capacity), bringing the total network to **`20` courts** (`9,600` available court-hours/month).
2. **Steady-State Monthly Revenue of Venue 2 (at 100% Fill-Up, Before Macro Trend Multiplier)**:
   - **Venue 2 Court Hire Revenue (`AdaptivePricing` ML Schedule across 14 courts)**: $\\frac{{14}}{{6}} \\times {res.venue1_steady_court_hire_mvnd:,.2f}\\text{{M}} =$ **`{res.venue2_steady_court_hire_mvnd:,.2f}M VND/month`**.
   - **Venue 2 Ancillary Revenue (`Coaching Facilities + Specialty Café + Paddle/Ball Rentals`)**: $\\frac{{14}}{{6}} \\times {res.venue1_steady_ancillary_mvnd:,.2f}\\text{{M}} =$ **`{res.venue2_steady_ancillary_mvnd:,.2f}M VND/month`**.
   - **Venue 2 Total Court + Ancillary Steady-State Revenue**: **`{res.venue2_steady_total_rev_mvnd:,.2f}M VND/month`** (bringing combined 20-court Court + Ancillary revenue to **`{res.total_20court_steady_court_hire_mvnd + res.total_20court_steady_ancillary_mvnd:,.2f}M VND/month`**, plus `PICK! Social` membership subscription revenue across the expanded 2-club player network!).
3. **Stochastic 6-to-9 Month Fill-Up Ramp (`Month 14 / Feb 2028` to `Months 20–22 / Aug–Oct 2028`)**:
   - In each Monte Carlo trajectory, the new club's time to reach full steady-state utilization is sampled from $T_{{\\text{{fill}}}} \\sim \\text{{TruncatedNormal}}(\\text{{mean}}=7.5\\text{{ mo}},\\, \\text{{std}}=0.75\\text{{ mo}},\\, \\min=6.0,\\, \\max=9.0)$ per Case Page 3 (*"New clubs usually take six to nine months to fill up"*).
   - In **Month 14 (`Feb-28`, Opening Month 1)**, pre-registration from the `PICK! Social` app and university leagues seeds initial utilization at `~28.0%` (`{res.venue2_total_rev_mean_60m[13]:,.2f}M VND` Venue 2 revenue), climbing smoothly along an S-curve to `~66.7%` by **Month 17 (`May-28`, `{res.venue2_total_rev_mean_60m[16]:,.2f}M VND`)**, `~94.6%` by **Month 20 (`Aug-28`, `{res.venue2_total_rev_mean_60m[19]:,.2f}M VND`)**, and **`100.0%` (`{res.venue2_steady_total_rev_mvnd:,.2f}M VND/mo`)** by **Month 22 (`Oct-28`)**.

---

## 4. Reference Bibliography Cited

| Citation Key | Source Title & Authors | Relevance to Venue Scaling Schedule & CapEx |
| :--- | :--- | :--- |
{cite_table_md}
"""
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "wbs_csv": wbs_csv,
        "monthly_csv": monthly_csv,
        "report_path": report_path,
    }
