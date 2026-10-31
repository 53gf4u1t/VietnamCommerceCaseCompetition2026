"""Schedule WBS, Even CapEx Split (7.0B VND / 4 Months = 1,750M/mo), and 6-to-9 Month Fill-Up Ramp Engine for VenueScalingTimeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from .config import VenueScalingConfig


@dataclass(frozen=True)
class ScheduledVenueTask:
    id: str
    stage_id: str
    stage_name: str
    stage_short_name: str
    stage_color: str
    stage_light_color: str
    name: str
    core_deliverable: str
    track: str
    start_week: float
    end_week: float
    duration_weeks: float
    duration_days: int
    start_date: date
    end_date: date
    dependencies: tuple[str, ...]
    is_critical_path: bool
    is_parallel_track: bool
    owner: str
    capex_mvnd: float
    citation_keys: tuple[str, ...]
    estimation_rationale: str


@dataclass(frozen=True)
class ScheduledVenueStage:
    stage_id: str
    name: str
    short_name: str
    color: str
    light_color: str
    start_week: float
    end_week: float
    span_weeks: float
    start_date: date
    end_date: date
    task_count: int
    stage_capex_mvnd: float
    budget_share_pct: float
    summary: str


@dataclass(frozen=True)
class ScheduledVenueMilestone:
    id: str
    name: str
    week: float
    target_date: date
    stage_id: str
    stage_color: str
    description: str


@dataclass(frozen=True)
class VenueScalingResults:
    build_start_date: date
    build_end_date: date
    open_date: date
    tasks: tuple[ScheduledVenueTask, ...]
    stages: tuple[ScheduledVenueStage, ...]
    milestones: tuple[ScheduledVenueMilestone, ...]
    wbs_df: pd.DataFrame
    monthly_60m_df: pd.DataFrame
    venue1_steady_court_hire_mvnd: float
    venue1_steady_ancillary_mvnd: float
    venue2_steady_court_hire_mvnd: float
    venue2_steady_ancillary_mvnd: float
    venue2_steady_total_rev_mvnd: float
    total_20court_steady_court_hire_mvnd: float
    total_20court_steady_ancillary_mvnd: float
    # 60-month arrays (Jan-27 to Dec-31)
    month_labels: tuple[str, ...]
    active_courts_60m: np.ndarray
    fitout_capex_60m: np.ndarray
    venue2_lease_60m: np.ndarray
    venue2_fill_factor_mean_60m: np.ndarray
    venue2_fill_factor_p05_60m: np.ndarray
    venue2_fill_factor_p95_60m: np.ndarray
    venue2_court_hire_mean_60m: np.ndarray
    venue2_ancillary_mean_60m: np.ndarray
    venue2_total_rev_mean_60m: np.ndarray
    venue2_total_rev_p05_60m: np.ndarray
    venue2_total_rev_p95_60m: np.ndarray
    sample_fill_paths_60m: np.ndarray
    sample_v2_rev_paths_60m: np.ndarray


def build_venue_scaling_schedule(
    cfg: VenueScalingConfig,
    adaptive_output_dir: Path | None = None,
) -> VenueScalingResults:
    """
    Build the Q3 2027 14-court venue expansion WBS schedule, compute the even 1,750.0M VND/month
    CapEx allocation across the 4 construction months (Jul-27 to Oct-27), and simulate the
    stochastic 6-to-9 month fill-up ramp (starting Month 11 / Nov-27) using OUR AdaptivePricing
    model rather than the Case's static 1.5B/month figure.
    """
    meta = cfg.meta
    base_date = date.fromisoformat(meta.build_start_date)

    # Load Venue 1 (6-court) ML Adaptive Court Hire Revenue from AdaptivePricing/outputs/
    if adaptive_output_dir is None:
        ap_dir = cfg.configs_dir.parent / "AdaptivePricing" / "outputs"
    else:
        ap_dir = adaptive_output_dir.resolve()

    daily_csv = ap_dir / "daily_revenue_comparison.csv"
    if not daily_csv.exists():
        raise FileNotFoundError(
            f"AdaptivePricing daily_revenue_comparison.csv not found at {daily_csv}. "
            "Run `uv run main` from CaseComp so AdaptivePricing executes first."
        )
    ap_df = pd.read_csv(daily_csv)
    v1_court_hire_mvnd = float(ap_df["ml_adaptive_rev_mvnd"].sum())
    v1_ancillary_mvnd = cfg.venue1_ancillary_rev_mvnd

    # Under OUR model:
    # Existing Venue 1 = 6 courts; New Venue 2 = 14 courts (14/6 = 2.3333x capacity; Total = 20 courts).
    # At 100% steady-state fill-up (after the 6-to-9 month fill-up ramp), the 14 new courts generate:
    #   Venue 2 Steady Court Hire = (14 / 6) * v1_court_hire_mvnd
    #   Venue 2 Steady Ancillary  = (14 / 6) * v1_ancillary_mvnd (Paddle/Balls + Coaching + Specialty Cafe)
    cap_ratio = meta.capacity_ratio_new_vs_existing  # 14.0 / 6.0 = 2.333333
    v2_steady_court_hire = v1_court_hire_mvnd * cap_ratio
    v2_steady_ancillary = v1_ancillary_mvnd * cap_ratio
    v2_steady_tot_rev = v2_steady_court_hire + v2_steady_ancillary

    scheduled_tasks: list[ScheduledVenueTask] = []
    for t in cfg.tasks:
        st = cfg.stages[t.stage_id]
        end_w = t.start_week + t.duration_weeks
        s_dt = base_date + timedelta(days=int(round(t.start_week * 7)))
        e_dt = base_date + timedelta(days=int(round(end_w * 7)) - 1)
        dur_days = int(round(t.duration_weeks * 7))
        scheduled_tasks.append(
            ScheduledVenueTask(
                id=t.id,
                stage_id=t.stage_id,
                stage_name=st.name,
                stage_short_name=st.short_name,
                stage_color=st.color,
                stage_light_color=st.light_color,
                name=t.name,
                core_deliverable=t.core_deliverable,
                track=t.track,
                start_week=t.start_week,
                end_week=end_w,
                duration_weeks=t.duration_weeks,
                duration_days=dur_days,
                start_date=s_dt,
                end_date=e_dt,
                dependencies=t.dependencies,
                is_critical_path=t.is_critical_path,
                is_parallel_track=t.is_parallel_track,
                owner=t.owner,
                capex_mvnd=t.capex_mvnd,
                citation_keys=t.citation_keys,
                estimation_rationale=t.estimation_rationale,
            )
        )

    scheduled_stages: list[ScheduledVenueStage] = []
    for s_id, st in cfg.stages.items():
        s_tasks = [t for t in scheduled_tasks if t.stage_id == s_id]
        min_w = min(t.start_week for t in s_tasks)
        max_w = max(t.end_week for t in s_tasks)
        s_dt = base_date + timedelta(days=int(round(min_w * 7)))
        e_dt = base_date + timedelta(days=int(round(max_w * 7)) - 1)
        s_capex = round(sum(t.capex_mvnd for t in s_tasks), 2)
        scheduled_stages.append(
            ScheduledVenueStage(
                stage_id=s_id,
                name=st.name,
                short_name=st.short_name,
                color=st.color,
                light_color=st.light_color,
                start_week=min_w,
                end_week=max_w,
                span_weeks=max_w - min_w,
                start_date=s_dt,
                end_date=e_dt,
                task_count=len(s_tasks),
                stage_capex_mvnd=s_capex,
                budget_share_pct=st.budget_share_pct,
                summary=st.summary,
            )
        )

    scheduled_milestones: list[ScheduledVenueMilestone] = []
    for m in cfg.milestones:
        st = cfg.stages[m.stage_id]
        m_dt = base_date + timedelta(days=int(round(m.week * 7)) - 1)
        scheduled_milestones.append(
            ScheduledVenueMilestone(
                id=m.id,
                name=m.name,
                week=m.week,
                target_date=m_dt,
                stage_id=m.stage_id,
                stage_color=st.color,
                description=m.description,
            )
        )

    wbs_rows = []
    for t in scheduled_tasks:
        wbs_rows.append(
            {
                "task_id": t.id,
                "stage_id": t.stage_id,
                "stage_name": t.stage_short_name,
                "task_name": t.name,
                "core_deliverable": t.core_deliverable,
                "engineering_track": t.track,
                "start_week": f"W{t.start_week + 1:.1f}",
                "end_week": f"W{t.end_week:.1f}",
                "duration_weeks": t.duration_weeks,
                "duration_days": t.duration_days,
                "start_date": t.start_date.isoformat(),
                "end_date": t.end_date.isoformat(),
                "dependencies": ", ".join(t.dependencies) if t.dependencies else "Q4 2027 Kickoff",
                "critical_path": "Yes" if t.is_critical_path else "Parallel Track",
                "owner": t.owner,
                "capex_mvnd": t.capex_mvnd,
                "citations": ", ".join(t.citation_keys),
                "estimation_rationale": t.estimation_rationale,
            }
        )
    wbs_df = pd.DataFrame(wbs_rows)

    # Simulate 60-Month (Jan-27 to Dec-31) CapEx, Lease, Court Capacity & 6-to-9 Month Fill-Up Ramp
    cal_months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    years_short = ("27", "28", "29", "30", "31")
    month_labels = [f"{cm}-{ys}" for ys in years_short for cm in cal_months]
    n_months = len(month_labels)  # 60

    rng = np.random.default_rng(cfg.random_seed + 101)
    n_sims = cfg.n_simulations

    # Sample fill-up duration T_fill ~ TruncatedNormal(mean=7.5, std=0.75, min=6.0, max=9.0) months
    fill_months_draws = meta.fill_up_duration_months_dist.sample(rng, size=n_sims)
    init_util_draws = meta.initial_open_utilization_fraction_dist.sample(rng, size=n_sims)

    build_start_idx = meta.build_start_month_index - 1  # 0-based: 6 = Jul-27 (Month 7)
    build_end_idx = meta.build_end_month_index - 1      # 0-based: 9 = Oct-27 (Month 10)
    open_idx = meta.open_month_index - 1                # 0-based: 10 = Nov-27 (Month 11)
    lease_start_idx = meta.lease_start_month_index - 1  # 0-based: 6 = Jul-27 (Month 7)

    active_courts_60m = np.zeros(n_months, dtype=int)
    fitout_capex_60m = np.zeros(n_months, dtype=float)
    venue2_lease_60m = np.zeros(n_months, dtype=float)
    fill_mat = np.zeros((n_sims, n_months), dtype=float)
    v2_court_rev_mat = np.zeros((n_sims, n_months), dtype=float)
    v2_anc_rev_mat = np.zeros((n_sims, n_months), dtype=float)
    v2_tot_rev_mat = np.zeros((n_sims, n_months), dtype=float)

    for m in range(n_months):
        if m < open_idx:
            active_courts_60m[m] = meta.existing_courts  # 6 courts in Months 1..10 (Jan-27..Oct-27)
        else:
            active_courts_60m[m] = meta.total_expanded_courts  # 20 courts in Month 11+ (Nov-27..Dec-31)

        if build_start_idx <= m <= build_end_idx:
            # Evenly split the 7,000.0M VND fit-out CapEx across the 4 construction months (1,750.0M/mo)
            fitout_capex_60m[m] = meta.even_monthly_fitout_capex_mvnd
        else:
            fitout_capex_60m[m] = 0.0

        if m >= lease_start_idx:
            venue2_lease_60m[m] = meta.new_venue_lease_monthly_mvnd  # 330.0M VND/mo from Jul-27 onward
        else:
            venue2_lease_60m[m] = 0.0

        if m < open_idx:
            fill_mat[:, m] = 0.0
        else:
            # Months since opening: k = 1 in Month 11 (Nov-27), k = 2 in Month 12 (Dec-27), etc.
            k_open = float(m - open_idx + 1)
            # Smooth S-curve / linear-logistic fill-up reaching 1.0 at k_open = T_fill (6 to 9 months)
            progress = np.clip((k_open - 1.0) / np.maximum(1.0, fill_months_draws - 1.0), 0.0, 1.0)
            # Smooth hermite S-curve (3p^2 - 2p^3) blended with linear for realistic club fill-up
            s_curve = 0.45 * progress + 0.55 * (3.0 * (progress ** 2) - 2.0 * (progress ** 3))
            f_k = np.clip(init_util_draws + (1.0 - init_util_draws) * s_curve, 0.0, 1.0)
            fill_mat[:, m] = f_k

        v2_court_rev_mat[:, m] = fill_mat[:, m] * v2_steady_court_hire
        v2_anc_rev_mat[:, m] = fill_mat[:, m] * v2_steady_ancillary
        v2_tot_rev_mat[:, m] = v2_court_rev_mat[:, m] + v2_anc_rev_mat[:, m]

    p_low = cfg.ci_low_percentile
    p_high = cfg.ci_high_percentile

    fill_mean = fill_mat.mean(axis=0)
    fill_p05 = np.percentile(fill_mat, p_low, axis=0)
    fill_p95 = np.percentile(fill_mat, p_high, axis=0)

    v2_court_mean = v2_court_rev_mat.mean(axis=0)
    v2_anc_mean = v2_anc_rev_mat.mean(axis=0)
    v2_tot_mean = v2_tot_rev_mat.mean(axis=0)
    v2_tot_p05 = np.percentile(v2_tot_rev_mat, p_low, axis=0)
    v2_tot_p95 = np.percentile(v2_tot_rev_mat, p_high, axis=0)

    v2_opex = cfg.venue2_opex
    m60_rows = []
    for m in range(n_months):
        if m < open_idx:
            v2_util_cost = 0.0
            v2_maint_cost = 0.0
            v2_staff_cost = 0.0
        else:
            # Utilities are 65% fixed base + 35% scaling with court utilization fill factor
            v2_util_cost = v2_opex.electricity_lighting_water_dist.mean * (0.65 + 0.35 * float(fill_mean[m]))
            v2_maint_cost = v2_opex.maintenance_and_insurance_dist.mean
            v2_staff_cost = v2_opex.staff_wages_dist.mean

        m60_rows.append(
            {
                "month_index": m + 1,
                "month_label": month_labels[m],
                "year_num": (m // 12) + 1,
                "active_courts_total": int(active_courts_60m[m]),
                "venue1_courts": meta.existing_courts,
                "venue2_active_courts": meta.new_venue_courts if m >= open_idx else 0,
                "fitout_capex_even_split_mvnd": round(float(fitout_capex_60m[m]), 2),
                "venue2_lease_mvnd": round(float(venue2_lease_60m[m]), 2),
                "venue2_utilities_mean_mvnd": round(v2_util_cost, 2),
                "venue2_maint_ins_mean_mvnd": round(v2_maint_cost, 2),
                "venue2_staff_wages_mean_mvnd": round(v2_staff_cost, 2),
                "venue2_fill_up_factor_mean": round(float(fill_mean[m]), 4),
                "venue2_fill_up_factor_p05": round(float(fill_p05[m]), 4),
                "venue2_fill_up_factor_p95": round(float(fill_p95[m]), 4),
                "venue2_court_hire_mean_mvnd": round(float(v2_court_mean[m]), 2),
                "venue2_ancillary_rev_mean_mvnd": round(float(v2_anc_mean[m]), 2),
                "venue2_total_rev_mean_mvnd": round(float(v2_tot_mean[m]), 2),
                "venue2_total_rev_p05_mvnd": round(float(v2_tot_p05[m]), 2),
                "venue2_total_rev_p95_mvnd": round(float(v2_tot_p95[m]), 2),
            }
        )
    monthly_60m_df = pd.DataFrame(m60_rows)

    build_end_dt = base_date + timedelta(days=int(round(meta.build_duration_weeks * 7)) - 1)
    open_dt = build_end_dt + timedelta(days=1)

    return VenueScalingResults(
        build_start_date=base_date,
        build_end_date=build_end_dt,
        open_date=open_dt,
        tasks=tuple(scheduled_tasks),
        stages=tuple(scheduled_stages),
        milestones=tuple(scheduled_milestones),
        wbs_df=wbs_df,
        monthly_60m_df=monthly_60m_df,
        venue1_steady_court_hire_mvnd=v1_court_hire_mvnd,
        venue1_steady_ancillary_mvnd=v1_ancillary_mvnd,
        venue2_steady_court_hire_mvnd=v2_steady_court_hire,
        venue2_steady_ancillary_mvnd=v2_steady_ancillary,
        venue2_steady_total_rev_mvnd=v2_steady_tot_rev,
        total_20court_steady_court_hire_mvnd=v1_court_hire_mvnd + v2_steady_court_hire,
        total_20court_steady_ancillary_mvnd=v1_ancillary_mvnd + v2_steady_ancillary,
        month_labels=tuple(month_labels),
        active_courts_60m=active_courts_60m,
        fitout_capex_60m=fitout_capex_60m,
        venue2_lease_60m=venue2_lease_60m,
        venue2_fill_factor_mean_60m=fill_mean,
        venue2_fill_factor_p05_60m=fill_p05,
        venue2_fill_factor_p95_60m=fill_p95,
        venue2_court_hire_mean_60m=v2_court_mean,
        venue2_ancillary_mean_60m=v2_anc_mean,
        venue2_total_rev_mean_60m=v2_tot_mean,
        venue2_total_rev_p05_60m=v2_tot_p05,
        venue2_total_rev_p95_60m=v2_tot_p95,
        sample_fill_paths_60m=fill_mat[:40].copy(),
        sample_v2_rev_paths_60m=v2_tot_rev_mat[:40].copy(),
    )
