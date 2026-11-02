"""Structured CSV and Schedule + Cost Estimation + Cash-Flow Allocation Markdown exporter for PickSocialProjectTimeline."""

from __future__ import annotations

from pathlib import Path

from .config import TimelineConfig
from .schedule_model import TimelineResults


def export_timeline_outputs(
    cfg: TimelineConfig,
    res: TimelineResults,
    chart_paths: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Export WBS CSV, Concurrency CSV, Monthly/Weekly Cost Allocation CSVs, and structured REPORT.md into outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    wbs_csv = output_dir / "project_schedule_wbs.csv"
    res.wbs_df.to_csv(wbs_csv, index=False)

    conc_csv = output_dir / "weekly_concurrency_profile.csv"
    res.weekly_concurrency_df.to_csv(conc_csv, index=False)

    monthly_cost_csv = output_dir / "monthly_cost_allocation.csv"
    res.monthly_cost_allocation_df.to_csv(monthly_cost_csv, index=False)

    weekly_cost_csv = output_dir / "weekly_cost_allocation.csv"
    res.weekly_cost_allocation_df.to_csv(weekly_cost_csv, index=False)

    platform_opex_csv = output_dir / "recurring_platform_opex.csv"
    res.platform_opex_summary_df.to_csv(platform_opex_csv, index=False)

    report_path = output_dir / "REPORT.md"

    # Build Stage Summary Table rows (with Schedule + Cost + Buffer)
    stage_table_rows = []
    for st in res.stage_summaries:
        phase_type = "Core Build (M1-M4)" if st.stage_id in ("S1", "S2", "S3", "S4") else "Migration & Ops (M5-M6+)"
        stage_table_rows.append(
            f"| **{st.stage_id}** | **{st.short_name}** | `W{st.start_week + 1:.1f}-W{st.end_week:.1f}` ({st.start_date.strftime('%b %d')} - {st.end_date.strftime('%b %d')}) | "
            f"**{st.span_weeks:.1f}w** | {st.sum_person_weeks:.2f} pw | **{st.base_total_cost_mvnd:.2f}M** | +{st.contingency_buffer_mvnd:.2f}M (+{st.effective_buffer_pct:.1f}%) | "
            f"**{st.buffered_total_cost_mvnd:.2f}M** | {phase_type} |"
        )
    stage_table_md = "\n".join(stage_table_rows)

    # Build Milestones Table rows
    ms_rows = []
    for m in res.milestones:
        ms_rows.append(
            f"| **{m.id}** | **{m.name}** | `Week {m.week:.1f}` ({m.target_date.strftime('%Y-%m-%d')}) | {m.stage_id} | {m.description} |"
        )
    ms_table_md = "\n".join(ms_rows)

    # Build Task-by-Task Cost Table rows
    task_cost_rows = []
    for t in res.tasks:
        all_cites = ", ".join(f"`[{c}]`" for c in t.cost_citation_keys)
        task_cost_rows.append(
            f"| **`{t.id}`** | {t.name} | `W{t.start_week + 1:.1f}-W{t.end_week:.1f}` ({t.duration_weeks:.1f}w) | {t.role_allocation} | "
            f"`{t.person_weeks:.2f}` | `{t.base_labor_cost_mvnd:.2f}M` | `{t.direct_infra_cost_mvnd:.2f}M` | **`{t.base_total_cost_mvnd:.2f}M`** | "
            f"`+{t.contingency_buffer_pct:.0f}%` (`+{t.contingency_buffer_mvnd:.2f}M`) | **`{t.buffered_total_cost_mvnd:.2f}M`** | {t.cost_rationale} ({all_cites}) |"
        )
    task_cost_table_md = "\n".join(task_cost_rows)

    # Build Monthly Cost Allocation Table rows
    monthly_alloc_rows = []
    for _, r in res.monthly_cost_allocation_df.iterrows():
        monthly_alloc_rows.append(
            f"| **{r['month_code']} ({r['month_label']})** | `{r['weeks_window']}` | "
            f"`{r['leveled_sprint_active_base_mvnd']:.2f}M` | `{r['leveled_milestone_holdback_release_mvnd']:.2f}M` | "
            f"**`{r['leveled_committed_base_mvnd']:.2f}M`** | `+{r['leveled_contingency_reserve_mvnd']:.2f}M` | "
            f"**`{r['leveled_max_buffered_total_mvnd']:.2f}M`** | `{r['recurring_platform_opex_mvnd']:.2f}M` | "
            f"**`{r['total_platform_cost_with_opex_base_mvnd']:.2f}M` / `{r['total_platform_cost_with_opex_buffered_mvnd']:.2f}M`** | "
            f"{r['disbursement_mechanism']} |"
        )
    monthly_alloc_table_md = "\n".join(monthly_alloc_rows)

    # Build Recurring Platform OpEx Table rows
    popex_table_rows = []
    for _, r in res.platform_opex_summary_df.iterrows():
        popex_table_rows.append(
            f"| **`{r['item_code']}`** | {r['expense_line']} | `{r['active_window']}` | `{r['dist_type']}` | "
            f"**`{r['monthly_mean_mvnd']:.2f}M`** ($\\pm {r['monthly_std_mvnd']:.2f}\\text{{M}}$, `[{r['monthly_min_mvnd']:.2f}M, {r['monthly_max_mvnd']:.2f}M]`) | "
            f"**`{r['annual_steady_state_mvnd']:.2f}M / yr`** | {r['source']} |"
        )
    popex_table_md = "\n".join(popex_table_rows)

    # Build Citations Table rows
    cite_rows = []
    for c_key, c in cfg.citations.items():
        cite_rows.append(
            f"| `[{c_key}]` | **{c.title}** - *{c.authors}* ({c.publication}) | {c.benchmark_used} |"
        )
    cite_table_md = "\n".join(cite_rows)

    total_pw = sum(t.person_weeks for t in res.tasks)
    core_pw = sum(t.person_weeks for t in res.tasks if t.stage_id in ("S1", "S2", "S3", "S4"))

    report_md = f"""# PICK! Social & Adaptive Booking Platform (Path C) - Engineering Schedule, Cost Estimation & Cash-Flow Allocation Report

**Project**: {cfg.project.name}  
**Organization**: {cfg.project.company}  
**Plan Kickoff Date**: `2027-01-04` (First Monday of January 2027)  
**Core Product Completion Target**: **4 Months / 16.0 Weeks** (`2027-01-04` to `2027-04-25`, Milestone `M4`)  
**Full Lifecycle Horizon Modeled**: **6 Months / 24.0 Weeks** (`2027-01-04` to `2027-06-20+`, including Legacy Migration & Continuous Ops)  
**Engineering Squad Assumption**: **{cfg.project.team_size_fte:.1f} FTE** (1 Tech Lead/Architect, 2 Full-Stack Engineers, 1 Product UI/UX & Frontend Engineer, 0.5 QA/DevOps Automation Engineer) operating on **2-week Agile sprints**.  
**Total 4-Month Core Build Cost (`Stages 1-4`)**: **`{res.core_build_base_cost_mvnd:.2f}M VND` Base** + **`{res.core_build_buffer_mvnd:.2f}M VND` Risk Buffer** = **`{res.core_build_buffered_cost_mvnd:.2f}M VND` Buffered Budget**  
**Total 6-Month Full Lifecycle Cost (`Stages 1-6`)**: **`{res.total_base_cost_mvnd:.2f}M VND` Base** + **`{res.total_buffer_mvnd:.2f}M VND` Risk Buffer (`+{res.weighted_buffer_pct:.2f}%`)** = **`{res.total_buffered_cost_mvnd:,.2f}M VND` Buffered Budget (`~1.09B VND`)**

---

## 1. Executive Summary: Schedule & Cost Architecture

To transition PICK! from manual Zalo/phone/spreadsheet court reservations to a production-grade **Social & Adaptive Booking Platform (Path C)** without disrupting daily operations across its 6 leased courts (`3,600` monthly customers, `7,600` player visits/month), our engineering and financial plan integrates two core disciplines:

1. **4-Month Core Engineering Build (`Weeks 1-16`: `Jan 4 - Apr 25, 2027`) + 2-Month Phased Migration & Continuous Ops (`Weeks 16-24+`: `Apr 26 - Jun 20, 2027+`)**:
   - **Stages 1-4 (`Months 1-4`)** deliver the complete, load-tested, UAT-approved platform by **Milestone `M4` (`Apr 25, 2027`)** for **`{res.core_build_base_cost_mvnd:.2f}M VND` Base (`{res.core_build_buffered_cost_mvnd:.2f}M VND` Buffered)**.
   - **Stages 5-6 (`Months 5-6+`)** execute legacy data ETL, a 2-week parallel shadow booking run, full cutover (`Milestone M5`), and 30-day post-launch hypercare + live ML pricing recalibration (`Milestone M6`) for **`{res.total_base_cost_mvnd - res.core_build_base_cost_mvnd:.2f}M VND` Base (`{res.total_buffered_cost_mvnd - res.core_build_buffered_cost_mvnd:.2f}M VND` Buffered)**.
2. **Pay-As-Needed Cash-Flow-Smoothed Cost Allocation (Zero Upfront Lump-Sum Shock)**:
   - Rather than paying a traditional agency `45%` upfront lump-sum deposit (`~490M VND` in Month 1) or suffering an unsmoothed Stage 3 accrual spike (`~239M VND/month` in Months 2-3), PICK! funds the build through **bi-weekly Agile sprint drawdowns**, a **`22%` Stage 3 milestone acceptance holdback** (released across `M4` UAT Sign-off in Month 4, `M5` Cutover in Month 5, and `M6` Hypercare Sign-off in Month 6), and a **conditional monthly contingency reserve (`29.58M VND/month`)**.
   - As a result, **Committed Base monthly cash outflow is held within `139.93M - 165.97M VND/month` (`151.86M VND/mo` average)**-100% self-funded by PICK!'s ongoing monthly operating cash flow (`160.00M VND/mo` Baseline / `185.75M VND/mo` with `AdaptivePricing`).

### 1.1 Stage-Level Schedule & Cost Summary

| Stage ID | Stage Name | Calendar Window (2027) | Span | Effort (Person-Wks) | Base Cost (M VND) | Risk Buffer (M VND) | Buffered Total (M VND) | Lifecycle Phase |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
{stage_table_md}
| **CORE** | **Subtotal 4-Month Core Build (`S1-S4`)** | **`W1.0-W16.0` (Jan 04 - Apr 25)** | **16.0w** | **{core_pw:.2f} pw** | **{res.core_build_base_cost_mvnd:.2f}M** | **+{res.core_build_buffer_mvnd:.2f}M (+{(res.core_build_buffer_mvnd / res.core_build_base_cost_mvnd) * 100:.1f}%)** | **{res.core_build_buffered_cost_mvnd:.2f}M** | **Core Build Target** |
| **TOTAL** | **Full 6-Month Lifecycle (`S1-S6`)** | **`W1.0-W24.0` (Jan 04 - Jun 20)** | **24.0w** | **{total_pw:.2f} pw** | **{res.total_base_cost_mvnd:.2f}M** | **+{res.total_buffer_mvnd:.2f}M (+{res.weighted_buffer_pct:.1f}%)** | **{res.total_buffered_cost_mvnd:,.2f}M** | **Full 6-Month Program** |

### 1.2 Key Decision Gates & Payment Release Milestones (`M1`-`M6`)

| ID | Milestone Name | Target Date | Stage | Gate Exit & Disbursement Criteria |
| :--- | :--- | :--- | :---: | :--- |
{ms_table_md}

---

## 2. How the 4-Month (16-Week) Core Schedule Target Was Derived

Our schedule combines **bottom-up SaaS feature sizing** with **top-down parametric phase distribution** from established software engineering economics:

1. **Baseline Sports Court Booking MVP (`12.0 Weeks`)**:
   - Industry benchmarks for a production multi-court booking platform (`[NextLevelSports2025]`) establish a **10 to 14 week (12-week median)** timeline for standard court availability grids, Redis slot-locking, payment checkout, admin dashboard, and UAT.
2. **Path C Custom Scope Delta (`+4.0 Weeks Net Schedule` via Parallel Track Execution)**:
   - PICK!'s Path C platform adds two proprietary capabilities beyond a generic booking tool:
     - **Real-Time `AdaptivePricing` ML Tariff API (`T3.2`)**: `2.5 weeks` of integration work (reduced from `6+ weeks` because the HistGBR demand model and 4x4 regime tariff table are already pre-built and validated in `AdaptivePricing/`).
     - **PICK! Social Hub (`T3.4` - ELO/DUPR Skill Rating, 4-Player Open-Play Matchmaking & Coaching Splits)**: `3.5 weeks` of full-stack engineering.
   - Because our **4.5 FTE squad** executes Backend/ML (`T3.2`), Payments (`T3.3`), and Social Matchmaking (`T3.4`) in overlapped 2-week sprints backed by continuous CI testing (`T3.6`), `6.0 task-weeks` of added scope compresses into **`+4.0 calendar weeks`**, yielding **16.0 weeks (4.0 months)** from kickoff to UAT completion (`M4`).
3. **Alignment with COCOMO II & McConnell Phase Distribution**:
   - Across the 16-week core build (`56.75 person-weeks` of engineering effort), our allocation matches **Barry Boehm's COCOMO II (`[Boehm2000]`)** and **Steve McConnell's (`[McConnell2006]`)** empirical phase ratios:
     - **Discovery, Architecture & Detailed Specs (`Stage 1`)**: `16.5%` of core effort (`3.5w` span) vs. `15-20%` COCOMO II Inception/Elaboration benchmark.
     - **Cloud Infrastructure, DevOps & Auth Foundation (`Stage 2`)**: `12.5%` of core effort (`2.5w` span) vs. `10-12%` DORA/Well-Architected foundation benchmark.
     - **Core Feature Construction + Simultaneous Sprint QA (`Stage 3`)**: `54.0%` of core effort (`8.0w` span, 4 sprints) vs. `50-55%` COCOMO II Construction + Unit Test benchmark.
     - **Final System Integration, Load/Security Hardening & UAT (`Stage 4`)**: `17.0%` of core effort (`3.0w` span) vs. `15-20%` McConnell / IEEE 1012 System V&V & Transition benchmark.

---

## 3. Stage-by-Stage Schedule Justification (How, Why & Citations)

Every task in the Work Breakdown Structure (`outputs/project_schedule_wbs.csv`) is justified below with (a) **how the duration was calculated**, (b) **why that time is technically required**, and (c) **the supporting benchmark citation**.

### Stage 1: Discovery, Architecture & Detailed Specs (`Weeks 1.0-3.5` | `Jan 4 - Jan 27, 2027`)

| Task ID & Name | Duration & Window | How We Got This Number (Calculation) | Why It Is Needed & Technical Scope | Cited Sources |
| :--- | :---: | :--- | :--- | :--- |
| **`T1.1` Product Discovery, Segment Mapping & Operational Specs** | **1.5 wks** (`7.5 workdays` / `W1.0-W2.5`) | `3 days` stakeholder/staff shadowing (front desk, coaches, manager) + `2.5 days` mapping user journeys for PICK!'s 4 segments (Competitive, Social, Trend Explorers, Families) + `2 days` PRD sign-off. | COCOMO II allocates `8-10%` of core schedule (`~1.5w` of `16w`) to requirement baselines. Skipping front-desk workflow mapping causes severe UAT rework (`10x` defect cost multiplier). | `[Boehm2000]`, `[McConnell2006]` |
| **`T1.2` System Architecture, ACID DB Schema & Redis Lock Design** | **1.5 wks** (`7.5 workdays` / `W2.0-W3.5`) | `3 days` C4 system architecture + `2.5 days` PostgreSQL ACID schema (users, courts, slots, orders, coach splits) + `2 days` Redis distributed lock (`SETNX`/TTL) state-machine spec. | Peak evening (`95%` util) and weekend (`90%` util) court slots experience simultaneous booking bursts. Designing Redis TTL hold locks + Postgres serializable constraints upfront prevents race-condition double-bookings. | `[Boehm2000]`, `[Kleppmann2017]` |
| **`T1.3` OpenAPI 3.1 Contracts, UI/UX Design System & Gateway KYB** | **2.0 wks** (`10 workdays` / `W2.5-W4.5`) | `5 days` mobile-first Figma wireframes & design system + `3 days` OpenAPI 3.1 REST/webhook schemas + `2 days` submitting corporate KYB/ERC documents to PayOS, VNPay & MoMo. | Freezing OpenAPI 3.1 contracts lets frontend and backend squads code in parallel during Stage 3. Submitting Vietnamese payment gateway KYB in Week 2 absorbs the `3-7 working day` merchant approval lead time before coding starts. | `[McConnell2006]`, `[VNPayPayOS2025]` |

### Stage 2: Cloud Infrastructure, DevOps & Core Foundation (`Weeks 3.0-5.5` | `Jan 25 - Feb 10, 2027`)

| Task ID & Name | Duration & Window | How We Got This Number (Calculation) | Why It Is Needed & Technical Scope | Cited Sources |
| :--- | :---: | :--- | :--- | :--- |
| **`T2.1` Cloud Infrastructure (IaC), PostgreSQL & Redis Provisioning** | **1.5 wks** (`7.5 workdays` / `W3.0-W4.5`) | `3 days` Terraform/Docker VPC & container orchestration + `2.5 days` managed PostgreSQL (multi-AZ backups) & Redis cluster setup + `2 days` Cloudflare CDN, TLS & secrets management. | Standardized IaC provisioning for staging and production environments takes `1-1.5 weeks` per AWS Well-Architected benchmarks and ensures environment parity before Sprint 1. | `[AWSWellArch2024]`, `[DORA2023]` |
| **`T2.2` CI/CD Pipelines, Automated Test Harness & Observability** | **1.5 wks** (`7.5 workdays` / `W3.5-W5.0`) | `3 days` GitHub Actions build/deploy pipelines + `2.5 days` Pytest (backend) & Playwright (E2E) test harness with ephemeral test DBs + `2 days` Sentry/Prometheus alerting. | DORA Elite engineering benchmarks show that investing `1.5 weeks` in automated CI test gates and preview environments cuts downstream integration bugs by `>40%`. | `[DORA2023]`, `[WQR2024]` |
| **`T2.3` Auth Service, Phone/Zalo OTP & Role-Based Access (RBAC)** | **1.5 wks** (`7.5 workdays` / `W4.0-W5.5`) | `3 days` JWT/OAuth2 session service + `2.5 days` Vietnam Zalo ZNS / SMS OTP phone verification + `2 days` 4-role RBAC middleware (Player, Coach, Front-Desk, Manager). | Vietnamese players predominantly authenticate via mobile phone / Zalo rather than email/password. Sports booking MVP benchmarks allocate `1.5 weeks` for auth + multi-role RBAC. | `[NextLevelSports2025]`, `[OWASP2024]` |

### Stage 3: Core Platform Development & Simultaneous Testing (`Weeks 5.0-13.0` | `Feb 8 - Apr 4, 2027`)

| Task ID & Name | Duration & Window | How We Got This Number (Calculation) | Why It Is Needed & Technical Scope | Cited Sources |
| :--- | :---: | :--- | :--- | :--- |
| **`T3.1` Real-Time Court Booking Engine & Redis Concurrency Locking** | **3.0 wks** (`1.5 sprints` / `W5.0-W8.0`) | `5 days` 6-court x 16-hour availability API & Redis `5-min` checkout hold lock + `5 days` interactive mobile calendar grid + `5 days` cancellation/rescheduling & waitlist auto-fill. | The booking grid is the transactional core of PICK!. Industry benchmarks allocate `3 weeks` to build and race-test atomic slot locking, recurring bookings, and conflict resolution. | `[NextLevelSports2025]`, `[Kleppmann2017]` |
| **`T3.2` AdaptivePricing ML Engine Integration & Dynamic Tariff API** | **2.5 wks** (`1.25 sprints` / `W6.5-W9.0`) | `4 days` FastAPI pricing microservice wrapping `AdaptivePricing` lookup tables + `4 days` dynamic UI price badges (off-peak green deals vs. prime slots) + `4.5 days` promo/bundle rules. | Because the HistGBR demand model (`461.89M VND/mo` court revenue across 3 anchored regime demand curves) is already developed in `AdaptivePricing/`, Stage 3 only needs API wrapping, caching, and UI price elasticity cues (`2.5 weeks`). | `[InternalAdaptivePricing]`, `[Boehm2000]` |
| **`T3.3` Payment Gateways (VietQR/PayOS/VNPay/MoMo) & F&B Pre-Order** | **3.0 wks** (`1.5 sprints` / `W7.5-W10.5`) | `5 days` PayOS/VietQR & VNPay/MoMo HMAC-SHA256 request signing + `5 days` idempotent IPN webhook handlers & auto-refund ledger + `5 days` paddle/ball rental & F&B add-on checkout cart. | Vietnam payment gateway benchmarks specify `1-2 weeks` for sandbox checkout + server-to-server IPN webhook idempotency (handling mobile browser drops & GMT+7 timestamps), plus `1 week` for ancillary F&B/rental upsell cart logic. | `[VNPayPayOS2025]`, `[SaigonTech2024]` |
| **`T3.4` PICK! Social Hub: Skill Rating, Open-Play Matching & Coaching** | **3.5 wks** (`1.75 sprints` / `W8.5-W12.0`) | `5 days` player skill rating (DUPR/ELO self + peer rating) + `6 days` "Find-a-Game" 4-player open-play lobby & split-court payment + `6.5 days` coach scheduling & `52.17%` fee-split tracking. | Social matchmaking is the primary retention driver converting `30%` Trend Explorers and `35%` Social Players into repeat weekday daytime bookings. Scoped at `3.5 weeks` across frontend and backend. | `[McConnell2006]`, `[NextLevelSports2025]` |
| **`T3.5` Admin Portal, Front-Desk QR Check-In & Financial Ledger** | **2.5 wks** (`1.25 sprints` / `W10.5-W13.0`) | `4 days` staff QR check-in scanner & walk-in POS override + `4 days` court lighting/maintenance block manager + `4.5 days` automated daily Revenue/COGS/OpEx CSV & P&L export. | Replaces manual spreadsheet tracking for front-desk staff (`130M VND/mo` payroll) and feeds standardized financial logs directly into `FinancialModel/`. | `[NextLevelSports2025]` |
| **`T3.6` [Simultaneous QA] Continuous Unit, API Contract & Sprint Testing** | **8.0 wks** (`All 4 Sprints` / `W5.0-W13.0`) | Runs **in parallel** across all 8 weeks of Stage 3 (`~25%` of engineering bandwidth): writing unit tests (`>80%` coverage), OpenAPI contract tests, Redis concurrency race tests, and Playwright UI tests inside every PR. | Capers Jones (`[CapersJones2012]`) and the *World Quality Report* (`[WQR2024]`) show defect removal takes `30-40%` of total effort. Embedding continuous testing inside Stage 3 sprints removes `>85%` of defects *before* Stage 4 UAT. | `[CapersJones2012]`, `[WQR2024]`, `[DORA2023]` |

### Stage 4: Final System Integration, Load/Security Testing & UAT (`Weeks 13.0-16.0` | `Apr 5 - Apr 25, 2027`)

| Task ID & Name | Duration & Window | How We Got This Number (Calculation) | Why It Is Needed & Technical Scope | Cited Sources |
| :--- | :---: | :--- | :--- | :--- |
| **`T4.1` End-to-End (E2E) Integration & Payment Gateway Certification** | **1.5 wks** (`7.5 workdays` / `W13.0-W14.5`) | `3 days` full cross-module E2E test suites + `3 days` formal VNPay/MoMo/PayOS technical compliance review (`2-3 working days` per gateway SLA) + `1.5 days` live 10,000 VND penny-test settlement. | Vietnamese payment providers require formal sandbox UAT verification (checking duplicate IPN webhooks, signature sorting, and refund callbacks) before issuing live production API keys. | `[VNPayPayOS2025]`, `[IEEE1012]` |
| **`T4.2` Peak Concurrency Load Testing (k6) & Security/PDPD Audit** | **1.5 wks** (`7.5 workdays` / `W13.5-W15.0`) | `3.5 days` k6/Locust stress testing (`500` concurrent users booking prime 18:00 slots simultaneously) + `2 days` OWASP ASVS & Vietnam Decree 13/2023/ND-CP (PDPD) audit + `2 days` remediation. | Validates `<150ms` p95 response time and **zero double-booked courts** under flash-booking load, while ensuring compliance with Vietnam's Personal Data Protection Decree. | `[OWASP2024]`, `[Kleppmann2017]` |
| **`T4.3` Staff Training & Pilot User Acceptance Testing (UAT)** | **1.5 wks** (`7.5 workdays` / `W14.5-W16.0`) | `2.5 days` on-site training for front-desk staff, coaches & manager + `3 days` live pilot UAT with `50` real players across all 4 segments + `2 days` final bug triage & Go-Live sign-off (`M4`). | Both McConnell (`[McConnell2006]`) and sports booking benchmarks (`[NextLevelSports2025]`) allocate `1.5-2 weeks` (`~10%` of build schedule) for real-world operational UAT at the venue. | `[McConnell2006]`, `[NextLevelSports2025]` |

### Stage 5: Legacy Booking Migration & Phased Cutover (`Weeks 16.0-20.0` | `Apr 26 - May 23, 2027`)

| Task ID & Name | Duration & Window | How We Got This Number (Calculation) | Why It Is Needed & Technical Scope | Cited Sources |
| :--- | :---: | :--- | :--- | :--- |
| **`T5.1` Legacy Data Audit, Cleansing & ETL (Zalo/Sheets to Postgres)** | **1.5 wks** (`7.5 workdays` / `W16.0-W17.5`) | `3 days` exporting & deduplicating `1,150` registered accounts + `3,600` monthly player phone numbers + `2.5 days` Python ETL scripts for prepaid coaching packages & future court blocks + `2 days` executing **2 staging dry runs**. | Kimball ETL lifecycle benchmarks (`[Kimball2013]`) mandate auditing unstructured spreadsheet/chat records and performing at least 2 full staging dry runs to guarantee zero lost prepaid balances. | `[Kimball2013]`, `[NextLevelSports2025]` |
| **`T5.2` Parallel Shadow Run & Priority Cohort Soft Launch** | **2.0 wks** (`14 calendar days` / `W17.0-W19.0`) | **2 full weekly operational cycles** (`2 x 5 weekdays + 2 x 2 weekend peaks`) running the new platform in parallel with front-desk Zalo support, onboarding Competitive (`20%`) and Social (`35%`) cohorts first. | Enterprise cutover standards require a `2-week` parallel run across two weekend peaks to verify `100%` ledger reconciliation between front-desk POS and online VietQR webhooks before retiring legacy booking. | `[Kimball2013]`, `[McConnell2006]` |
| **`T5.3` Legacy Booking Cutover & 100% Self-Service Public Launch** | **1.0 wk** (`7 calendar days` / `W19.0-W20.0`) | `1 night` (Sunday 22:30 off-peak window) final delta migration + `3 days` switching Zalo Official Account & social bio links (`42,000` followers) to direct self-service web booking + `3 days` floor-staff assisted check-in. | Executing final cutover during Sunday night off-peak hours (`[NextLevelSports2025]`) ensures zero court downtime, completing `100%` migration by **Milestone `M5` (`May 23, 2027`)**. | `[NextLevelSports2025]` |

### Stage 6: Post-Launch Feedback Loop, Continuous Development & Patching (`Weeks 19.0-24.0+` | `May 17 - Jun 20, 2027+`)

| Task ID & Name | Duration & Window | How We Got This Number (Calculation) | Why It Is Needed & Technical Scope | Cited Sources |
| :--- | :---: | :--- | :--- | :--- |
| **`T6.1` Post-Launch Hypercare, SLA Monitoring & Hotfix Patching** | **3.0 wks** (`21 calendar days` / `W19.0-W22.0`) | Covers **3 full weekly peak cycles** post-cutover with dedicated on-call engineering during prime hours (`17:00-22:00` & weekends) and `<2 hour` hotfix deployment SLA via CI/CD. | ITIL v4 Early Life Support (`[ITILv4]`) and DORA benchmarks (`[DORA2023]`) prescribe a `3-4 week` hypercare window to triage real-world mobile browser quirks, OTP delivery delays, and webhook retries. | `[ITILv4]`, `[DORA2023]` |
| **`T6.2` User Telemetry Analysis, UX Tuning & ML Pricing Calibration** | **3.5 wks** (`24.5 days` / `W20.5-W24.0`) | `3.0 weeks` (1 full 30-day post-cutover billing cycle) collecting unconfounded live price-elasticity and open-play conversion telemetry + `0.5 weeks` retraining `AdaptivePricing` HistGBR parameters. | Kohavi et al. (`[Kohavi2020]`) establish that online behavioral and pricing experiments require a minimum `3-4 week` window to capture full day-of-week and monthly payday cycles without novelty distortion. | `[Kohavi2020]`, `[InternalAdaptivePricing]` |
| **`T6.3` Continuous Feature Releases (Tournaments, Loyalty) & CI/CD** | **3.0+ wks** (`Ongoing Cadence` / `W21.0-W24.0+`) | Bi-weekly `2-week` Agile release sprints delivering post-MVP retention features (automated tournament brackets, loyalty points, push notifications) with automated regression suites. | Sustains long-term product velocity and keeps platform maintenance within the `SG&A: Booking system & administration` budget (`10.0M VND/mo`). | `[DORA2023]` |

---

## 4. Cost Estimation Methodology, HCMC Rate Card & Risk Buffer Architecture

### 4.1 Loaded Vietnam (HCMC) Engineering Rate Card

To derive realistic, bottom-up engineering costs in Ho Chi Minh City, we combine **ITviec & TopDev Vietnam IT Salary Benchmark Reports (`[ITviecTopDev2025]`)** with mandatory **Vietnam Labor Code (`[VietnamLaborCode2019]`)** employer contributions (`17.5%` Social Insurance BHXH + `3.0%` Health Insurance BHYT + `1.0%` Unemployment Insurance BHTN = **`21.5%` statutory employer on-cost** plus cloud IDE/workstation overhead):

| Engineering Role | Squad FTE | HCMC Gross Base Salary (`M VND/mo`) | Statutory Employer On-Cost & Tooling (`+21.5%`) | Loaded Monthly Rate (`M VND/mo`) | Loaded Weekly Rate (`M VND / person-wk`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tech Lead / Solutions Architect** | `1.0 FTE` | `54.32M` | `+11.68M` | **`66.00M`** | **`16.50M / pw`** |
| **Senior / Mid Full-Stack Engineer** | `2.0 FTE` | `39.51M` | `+8.49M` | **`48.00M`** | **`12.00M / pw`** |
| **QA Automation / DevOps SDET** | `0.5 FTE` | `36.21M` | `+7.79M` | **`44.00M`** | **`11.00M / pw`** |
| **Product UI/UX & Frontend Engineer** | `1.0 FTE` | `32.92M` | `+7.08M` | **`40.00M`** | **`10.00M / pw`** |

### 4.2 Task Risk Contingency Buffer Calibration (`+15%` to `+25%`)

Following **PMI PMBOK 7th Edition (`[PMBOK7th]`)** and **COCOMO II (`[Boehm2000]`)** cost estimation standards, every task includes an explicit **Risk Contingency Buffer** above its expected Base Cost so PICK! is protected against scope discovery, third-party API delays, and peak-load edge cases:

- **Tier 1 - Low/Standard Technical Risk (`+15%` Buffer)**: Applied to internal discovery (`T1.1`), standard PostgreSQL/Redis schema design (`T1.2`), IaC/CI-CD templates (`T2.1`, `T2.2`, `T2.3`), internal Admin POS (`T3.5`), cutover execution (`T5.3`), and steady-state CI/CD feature sprints (`T6.3`).
- **Tier 2 - Moderate Multi-Module / Concurrency Risk (`+20%` Buffer)**: Applied to external payment KYB & OpenAPI spec alignment (`T1.3`), Redis distributed lock booking engine (`T3.1`), ML dynamic tariff API wrapping (`T3.2`), continuous multi-sprint QA (`T3.6`), payment certification (`T4.1`), venue UAT (`T4.3`), shadow run (`T5.2`), and post-launch hypercare/telemetry (`T6.1`, `T6.2`).
- **Tier 3 - High External Integration / Unstructured Data Risk (`+25%` Buffer)**: Applied to multi-gateway VietQR/VNPay/MoMo webhook idempotency (`T3.3`), 4-player social matchmaking & split-court payment state machines (`T3.4`), 500-user peak concurrency stress testing & OWASP/PDPD remediation (`T4.2`), and dirty legacy Zalo/Excel data cleansing (`T5.1`).

### 4.3 Task-by-Task Cost Estimation & Buffer Derivation (`T1.1`-`T6.3`)

| Task ID | Task Name | Window & Duration | Role Allocation | Effort (`pw`) | Base Labor (`M VND`) | Direct Cloud/Infra (`M VND`) | Base Total (`M VND`) | Risk Buffer (`%` / `M VND`) | Buffered Total (`M VND`) | Derivation & Cited Sources |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{task_cost_table_md}
| **TOTAL** | **All 21 WBS Tasks (`Stages 1-6`)** | **`W1.0-W24.0` (`24.0w`)** | **4.5 FTE Squad** | **`{total_pw:.2f}`** | **`{res.total_base_labor_cost_mvnd:.2f}M`** | **`{res.total_direct_infra_cost_mvnd:.2f}M`** | **`{res.total_base_cost_mvnd:.2f}M`** | **`+{res.weighted_buffer_pct:.2f}%` (`+{res.total_buffer_mvnd:.2f}M`)** | **`{res.total_buffered_cost_mvnd:,.2f}M`** | **Full 6-Month Program Budget** |

---

## 5. Pay-As-Needed Cost Allocation & Recurring Monthly Platform Operating Expenses (`Jan-27` to `Jun-27+`)

### 5.1 Three-Pillar Pay-As-Needed Cost Allocation Mechanism (`[PMBOK7th]`)

To spread the platform development investment as evenly as possible across the 6-month project duration (`Jan-27` to `Jun-27`) and pay **only when truly needed**:
1. **Pillar 1 - Bi-Weekly Sprint Earned-Value Drawdowns (`0%` Upfront Lump-Sum)**: Labor and direct cloud expenses are disbursed at the close of each 2-week sprint strictly for work packages active in that sprint. Cloud/hardware items (`64.44M VND` total) are paid only in the exact week provisioned (e.g., QR scanners in `W11-W12`, k6 load generators in `W14`, cutover SMS blasts in `W19`).
2. **Pillar 2 - `22.0%` Stage 3 Milestone Acceptance Holdback (`79.81M VND` Deferred Across M4, M5 & M6 Gates)**: Across Stage 3 tasks (`T3.1`-`T3.6`, Base Cost = `362.76M VND`), `78.0%` (`282.95M VND`) is paid during active Stage 3 sprints (`Months 2-3`), while **`22.0%` (`79.81M VND`) is held back** until formal quality and operational gates are passed:
   - **`15%` of Holdback (`11.97M VND`)** is released in **Month 4 (`Apr-27`, Weeks 15-16)** upon passing 500-user load testing and Milestone `M4` UAT Sign-Off.
   - **`45%` of Holdback (`35.91M VND`)** is released in **Month 5 (`May-27`, Weeks 19-20)** upon completing the 2-week parallel shadow run and Milestone `M5` 100% Legacy Cutover.
   - **`40%` of Holdback (`31.92M VND`)** is released in **Month 6 (`Jun-27`, Weeks 23-24)** upon completing the 30-day post-launch hypercare SLA and Milestone `M6` Live ML Recalibration Sign-Off.
3. **Pillar 3 - Even Monthly Contingency Reserve Allocation (`29.58M VND/month`)**: Rather than paying buffers upfront, the `177.46M VND` risk buffer is held as an even **`29.58M VND/month` (`7.39M VND/week`) conditional reserve** across `Jan-27` to `Jun-27` and drawn only if a task triggers its risk contingency.

### 5.2 Recurring Monthly Operating Expenses of the Live Platform (`Month 5 / May 2027+`)

Once the 4-month core build completes at the end of Month 4 (`Apr 25, 2027`) and the platform goes live in **Month 5 (`May 2027`)**, PICK! incurs recurring monthly operating expenses to run and maintain the digital ecosystem (configured as distributions in `configs/opex.toml`):

| Code | Recurring Platform Expense Line | Active Window | Distribution | Monthly Mean ($\\pm$ SD, `[Min, Max]`) | Annual Steady-State | Derivation & Cited Sources |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
{popex_table_md}
| **M5-M6** | **Launch & Hypercare Monthly Platform OpEx (`May-Jun 2027`)** | `Months 5-6` | Sum of `POPEX_1..4` | **`{res.launch_monthly_platform_opex_mvnd:.2f}M / mo`** | - | Engineering maintenance covered by active Stage 6 squad |
| **M7+** | **Steady-State Monthly Platform OpEx (`Jul 2027 - Dec 2031`)** | `Months 7-60` | Sum of `POPEX_1..5` | **`{res.steady_state_monthly_platform_opex_mvnd:.2f}M / mo`** | **`{res.steady_state_monthly_platform_opex_mvnd * 12.0:.2f}M / yr`** | Full cloud, OTP, SaaS, university ops & maintenance retainer |

### 5.3 Monthly Cost Allocation & Platform OpEx Schedule (`Jan-27` to `Jun-27`)

| Month (2027) | Weeks | Leveled Sprint Active Base (`M VND`) | Milestone Holdback Release (`M VND`) | **Leveled Dev Committed Base (`M VND`)** | Conditional Contingency Reserve (`M VND`) | **Leveled Dev Max Buffered (`M VND`)** | Recurring Platform OpEx (`M VND`) | **Total Monthly Platform Outflow (`Base / Buf`)** | Pay-As-Needed Disbursement Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{monthly_alloc_table_md}
| **6-Mo Total** | **`W1-W24`** | **`831.37M`** | **`79.81M`** | **`911.18M`** | **`+177.46M`** | **`1,088.64M`** | **`33.00M`** | **`944.18M / 1,121.64M`** | **Passed directly to `FinancialModel/`** |
| **Ongoing M7+** | **`Jul-27+`** | **`0.00M`** | **`0.00M`** | **`0.00M`** | **`0.00M`** | **`0.00M`** | **`26.50M / mo`** | **`26.50M / mo`** | **Steady-state cloud, OTP, uni ops & maintenance** |

---

## 6. Reference Bibliography & Industry Benchmarks Cited

| Citation Key | Reference Source & Publication | Specific Estimation Benchmark Applied |
| :--- | :--- | :--- |
{cite_table_md}

---

## 7. Generated Deliverables in `PickSocialProjectTimeline/outputs/`

- **Master Production Gantt Chart & WBS Cost Table (`.png`)**: `outputs/01_pick_social_platform_gantt_chart.png`
- **Stage Effort vs. Benchmarks & Weekly Concurrency Chart (`.png`)**: `outputs/02_stage_effort_and_concurrency_profile.png`
- **Stage Cost Breakdown & Leveled Monthly Cost Allocation Chart (`.png`)**: `outputs/03_cost_estimation_and_cashflow_allocation.png`
- **Full Work Breakdown Structure, Task Cost & Buffer Citations CSV (`.csv`)**: `outputs/project_schedule_wbs.csv`
- **Monthly Leveled Cost Allocation & Platform OpEx CSV (`.csv`)**: `outputs/monthly_cost_allocation.csv`
- **Weekly Cost Allocation & Holdback Release CSV (`.csv`)**: `outputs/weekly_cost_allocation.csv`
- **Recurring Monthly Platform Operating Expenses CSV (`.csv`)**: `outputs/recurring_platform_opex.csv`
- **Weekly Active Workstream Concurrency CSV (`.csv`)**: `outputs/weekly_concurrency_profile.csv`
- **Structured Schedule, Cost & Allocation Methodology Report (`.md`)**: `outputs/REPORT.md`
"""
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "wbs_csv": wbs_csv,
        "conc_csv": conc_csv,
        "monthly_cost_csv": monthly_cost_csv,
        "weekly_cost_csv": weekly_cost_csv,
        "platform_opex_csv": platform_opex_csv,
        "report_path": report_path,
    }
