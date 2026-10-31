"""Core 3-statement financial model engine for PICK! integrating AdaptivePricing, PickSocialProjectTimeline, VenueScalingTimeline & External Factors (`externals.toml`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .config import FinancialModelConfig

REQUIRED_ADAPTIVE_PRICING_REPORTS = (
    "daily_revenue_comparison.csv",
    "monthly_calendar_pricing_schedule.csv",
    "REPORT.md",
    "ML_METRICS.md",
)

REQUIRED_TIMELINE_REPORTS = (
    "monthly_cost_allocation.csv",
    "recurring_platform_opex.csv",
    "project_schedule_wbs.csv",
    "REPORT.md",
)

REQUIRED_VENUE_SCALING_REPORTS = (
    "venue_scaling_wbs.csv",
    "monthly_expansion_capex_and_ramp.csv",
    "REPORT.md",
)

REQUIRED_MARKETING_REPORTS = (
    "monthly_segment_marketing_and_visits_60m.csv",
    "segment_ancillary_revenue_60m.csv",
    "REPORT.md",
    "ML_METRICS.md",
)


@dataclass(frozen=True)
class CourtRegimeMetrics:
    regime_name: str
    days_per_month: int
    hours_per_day: int
    available_court_hours: float
    case_utilisation_rate: float
    case_booked_hours: float
    case_price_kvnd: float
    case_revenue_mvnd: float
    ml_utilisation_rate: float
    ml_booked_hours: float
    ml_avg_price_kvnd: float
    ml_revenue_mvnd: float


@dataclass(frozen=True)
class AdaptivePricingIntegration:
    output_dir: Path
    verified_files: tuple[str, ...]
    sim_baseline_court_hire_mvnd: float
    case_baseline_court_hire_mvnd: float
    ml_adaptive_court_hire_mvnd: float
    lift_vs_case_mvnd: float
    lift_vs_case_pct: float
    lift_vs_sim_baseline_mvnd: float
    lift_vs_sim_baseline_pct: float
    sim_baseline_booked_hours: float
    ml_adaptive_booked_hours: float
    ml_blended_util_pct: float


@dataclass(frozen=True)
class TimelineCostIntegration:
    output_dir: Path
    verified_files: tuple[str, ...]
    monthly_committed_dev_base_6m: np.ndarray     # 6 months (Jan-27..Jun-27)
    monthly_contingency_reserve_6m: np.ndarray    # 6 months (Jan-27..Jun-27)
    monthly_max_buffered_dev_6m: np.ndarray       # 6 months (Jan-27..Jun-27)
    total_committed_dev_base_mvnd: float          # 911.18M
    total_contingency_reserve_mvnd: float         # 177.46M
    total_max_buffered_dev_mvnd: float            # 1,088.64M
    launch_platform_opex_m5_m6_mvnd: float        # 16.50M/mo
    steady_state_platform_opex_m7_plus_mvnd: float # 26.50M/mo


@dataclass(frozen=True)
class VenueScalingIntegration:
    output_dir: Path
    verified_files: tuple[str, ...]
    active_courts_60m: np.ndarray
    fitout_capex_even_split_60m: np.ndarray
    venue2_lease_60m: np.ndarray
    venue2_fill_factor_mean_60m: np.ndarray
    total_fitout_capex_mvnd: float
    even_monthly_fitout_capex_mvnd: float
    investor_equity_injection_mvnd: float
    investor_reserve_mvnd: float


@dataclass(frozen=True)
class MarketingIntegration:
    output_dir: Path
    verified_files: tuple[str, ...]
    followers_ml_60m: np.ndarray
    follower_to_customer_conv_rate_60m: np.ndarray
    customers_total_ml_60m: np.ndarray
    visits_competitive_ml_60m: np.ndarray
    visits_social_ml_60m: np.ndarray
    visits_trend_explorers_ml_60m: np.ndarray
    visits_families_ml_60m: np.ndarray
    visits_total_ml_60m: np.ndarray
    mkt_budget_competitive_ml_60m: np.ndarray
    mkt_budget_social_ml_60m: np.ndarray
    mkt_budget_trend_explorers_ml_60m: np.ndarray
    mkt_budget_families_ml_60m: np.ndarray
    mkt_budget_total_ml_60m: np.ndarray
    paddle_balls_total_ml_60m: np.ndarray
    coaching_total_ml_60m: np.ndarray
    food_drinks_total_ml_60m: np.ndarray
    ancillary_total_ml_60m: np.ndarray


@dataclass(frozen=True)
class MonteCarloSubscriptionSummary:
    n_simulations: int
    n_train: int
    n_test: int
    # External trend index I_t & adjusted visits (60 months)
    trend_index_mean: np.ndarray
    trend_index_p05: np.ndarray
    trend_index_p95: np.ndarray
    adjusted_visits_mean: np.ndarray
    adjusted_visits_p05: np.ndarray
    adjusted_visits_p95: np.ndarray
    # 60-month trajectory arrays (mean, P05, P95 across N simulations)
    unique_customers_mean: np.ndarray
    unique_customers_p05: np.ndarray
    unique_customers_p95: np.ndarray
    social_followers_mean: np.ndarray
    active_subscribers_mean: np.ndarray
    active_subscribers_p05: np.ndarray
    active_subscribers_p95: np.ndarray
    Active_subscribers_train_mean: np.ndarray
    active_subscribers_test_mean: np.ndarray
    new_subscribers_mean: np.ndarray
    churned_subscribers_mean: np.ndarray
    # Venue 2 (14 courts) fill-up factor & revenue arrays
    venue2_fill_factor_mean: np.ndarray
    venue2_fill_factor_p05: np.ndarray
    venue2_fill_factor_p95: np.ndarray
    venue2_total_rev_mean: np.ndarray
    # Revenue, Costs, EBITDA & Cash arrays
    membership_rev_mean: np.ndarray
    membership_rev_p05: np.ndarray
    membership_rev_p95: np.ndarray
    total_rev_mean: np.ndarray
    total_rev_p05: np.ndarray
    total_rev_p95: np.ndarray
    total_costs_mean: np.ndarray
    total_costs_p05: np.ndarray
    total_costs_p95: np.ndarray
    ebitda_mean: np.ndarray
    ebitda_p05: np.ndarray
    ebitda_p95: np.ndarray
    ebitda_train_mean: np.ndarray
    ebitda_test_mean: np.ndarray
    # Monthly Cash Balance distribution across N simulations
    ending_cash_mean: np.ndarray
    ending_cash_p05: np.ndarray
    ending_cash_p95: np.ndarray
    stress_ending_cash_mean: np.ndarray
    # Stochastic Case Baseline (Status Quo: 6 courts, static pricing, no platform, no Venue 2) across N simulations
    base_total_rev_mean: np.ndarray
    base_total_rev_p05: np.ndarray
    base_total_rev_p95: np.ndarray
    base_total_costs_mean: np.ndarray
    base_total_costs_p05: np.ndarray
    base_total_costs_p95: np.ndarray
    base_ebitda_mean: np.ndarray
    base_ebitda_p05: np.ndarray
    base_ebitda_p95: np.ndarray
    base_ending_cash_mean: np.ndarray
    base_ending_cash_p05: np.ndarray
    base_ending_cash_p95: np.ndarray
    base_annual_rev_mean: np.ndarray
    base_annual_rev_p05: np.ndarray
    base_annual_rev_p95: np.ndarray
    base_annual_costs_mean: np.ndarray
    base_annual_costs_p05: np.ndarray
    base_annual_costs_p95: np.ndarray
    base_annual_ebitda_mean: np.ndarray
    base_annual_ebitda_p05: np.ndarray
    base_annual_ebitda_p95: np.ndarray
    annual_rev_p05: np.ndarray
    annual_rev_p95: np.ndarray
    annual_costs_p05: np.ndarray
    annual_costs_p95: np.ndarray
    annual_ebitda_p05: np.ndarray
    annual_ebitda_p95: np.ndarray
    stress_annual_ebitda_p05: np.ndarray
    stress_annual_ebitda_p95: np.ndarray
    # Stress-test (-30% demand) 60-month trajectory arrays
    stress_active_subscribers_mean: np.ndarray
    stress_total_rev_mean: np.ndarray
    stress_total_costs_mean: np.ndarray
    stress_ebitda_mean: np.ndarray
    # Sampled spaghetti trajectories (first 40 sims x 60 months) for plotting
    sample_subscribers_paths: np.ndarray
    sample_total_rev_paths: np.ndarray
    sample_ebitda_paths: np.ndarray
    sample_cash_paths: np.ndarray
    sample_trend_paths: np.ndarray
    sample_base_rev_paths: np.ndarray
    sample_base_ebitda_paths: np.ndarray
    sample_base_cash_paths: np.ndarray
    # Key milestone subscriber & revenue metrics
    m5_launch_subscribers: float
    m12_end_y1_subscribers: float
    m24_end_y2_subscribers: float
    m36_end_y3_subscribers: float
    m60_end_y5_subscribers: float
    m24_end_y2_membership_rev_mvnd: float
    y3_annual_ebitda_mean_mvnd: float
    y3_annual_ebitda_p05_mvnd: float
    y3_annual_ebitda_p95_mvnd: float
    y3_annual_ebitda_stress_mvnd: float
    prob_exceed_investor_3500m_y3_pct: float
    subscriber_cohort_df: pd.DataFrame


@dataclass(frozen=True)
class LineItemRecord:
    section: str          # "REVENUE", "COGS", "OPEX", "SGA", "SUMMARY"
    template_sheet: str   # "Revenue", "COGS", "SG&A", "FS-Month / FS-Annual"
    template_category: str
    line_item: str
    behavior: str
    case_baseline_monthly_mvnd: float
    m1_build_monthly_mvnd: float
    m5_launch_monthly_mvnd: float
    m11_open20_monthly_mvnd: float
    m24_end_y2_monthly_mvnd: float
    five_year_avg_monthly_mvnd: float
    five_year_total_mvnd: float
    pct_of_5y_revenue: float
    notes: str


@dataclass(frozen=True)
class FinancialModelResults:
    adaptive: AdaptivePricingIntegration
    timeline_cost: TimelineCostIntegration
    venue_scaling: VenueScalingIntegration
    marketing: MarketingIntegration
    mc: MonteCarloSubscriptionSummary
    regimes: tuple[CourtRegimeMetrics, ...]
    total_available_court_hours: float
    total_booked_court_hours: float
    blended_utilisation_rate: float
    blended_court_rate_kvnd: float

    # Customer metrics
    unique_customers: int
    exact_player_visits: float
    reported_player_visits: int
    weighted_avg_spend_per_visit_kvnd: float

    # Case Static Baseline Headline Totals (Monthly M VND)
    case_court_hire_mvnd: float
    case_total_revenue_mvnd: float
    case_total_cogs_mvnd: float
    case_gross_profit_mvnd: float
    case_total_opex_mvnd: float
    case_total_sga_mvnd: float
    case_total_costs_mvnd: float
    case_ebitda_mvnd: float
    case_net_income_mvnd: float

    # Active Model Headline Totals (Monthly Average & 5Y Totals across 60 Months)
    court_hire_mvnd: float
    venue1_court_hire_m1_mvnd: float
    venue2_court_hire_steady_mvnd: float
    paddle_balls_mvnd: float
    coaching_mvnd: float
    food_drinks_mvnd: float
    m24_membership_rev_mvnd: float
    five_year_membership_rev_mvnd: float

    # 60-Month Mean Arrays for each line item
    monthly_active_courts_arr: np.ndarray
    monthly_trend_index_arr: np.ndarray
    monthly_venue1_court_hire_arr: np.ndarray
    monthly_venue2_court_hire_arr: np.ndarray
    monthly_court_hire_arr: np.ndarray
    monthly_membership_rev_arr: np.ndarray
    monthly_paddle_balls_arr: np.ndarray
    monthly_coaching_arr: np.ndarray
    monthly_food_drinks_arr: np.ndarray
    monthly_total_rev_arr: np.ndarray

    monthly_coach_fees_arr: np.ndarray
    monthly_fnb_supplies_arr: np.ndarray
    monthly_payment_consumables_arr: np.ndarray
    monthly_membership_event_cogs_arr: np.ndarray
    monthly_total_cogs_arr: np.ndarray
    monthly_gross_profit_arr: np.ndarray
    monthly_gross_margin_pct_arr: np.ndarray

    monthly_venue1_lease_arr: np.ndarray
    monthly_venue2_lease_arr: np.ndarray
    monthly_venue_lease_arr: np.ndarray
    monthly_utilities_arr: np.ndarray
    monthly_maint_ins_arr: np.ndarray
    monthly_total_opex_arr: np.ndarray

    monthly_venue1_staff_wages_arr: np.ndarray
    monthly_venue2_staff_wages_arr: np.ndarray
    monthly_staff_wages_arr: np.ndarray
    monthly_booking_admin_arr: np.ndarray
    monthly_marketing_arr: np.ndarray
    monthly_platform_dev_alloc_arr: np.ndarray
    monthly_platform_recurring_opex_arr: np.ndarray
    monthly_external_legal_safety_sga_arr: np.ndarray
    monthly_total_sga_arr: np.ndarray

    monthly_total_opex_and_sga_arr: np.ndarray
    monthly_total_costs_arr: np.ndarray
    monthly_ebitda_arr: np.ndarray
    monthly_ebitda_margin_pct_arr: np.ndarray
    monthly_net_income_arr: np.ndarray

    # 5-Year Cumulative Totals (M VND)
    five_year_revenue_mvnd: float
    five_year_cogs_mvnd: float
    five_year_gross_profit_mvnd: float
    five_year_opex_mvnd: float
    five_year_sga_mvnd: float
    five_year_total_costs_mvnd: float
    five_year_ebitda_mvnd: float
    five_year_net_income_mvnd: float

    # Annual Totals Arrays (5 Years: FY27..FY31)
    annual_revenue_arr: np.ndarray
    annual_total_costs_arr: np.ndarray
    annual_ebitda_arr: np.ndarray
    annual_net_income_arr: np.ndarray

    # Balance Sheet & Cash Flow Trajectories (60 months)
    starting_cash_mvnd: float
    min_cash_balance_model_mvnd: float
    min_cash_month_label: str
    m6_pre_expansion_cash_mvnd: float
    m10_post_build_cash_mvnd: float
    ending_cash_60m_model_mvnd: float
    ending_cash_60m_case_mvnd: float
    monthly_starting_cash_model: np.ndarray
    monthly_ocf_model: np.ndarray
    monthly_icf_capex_model: np.ndarray
    monthly_fcf_equity_model: np.ndarray
    monthly_net_cash_flow_model: np.ndarray
    monthly_ending_cash_model: np.ndarray
    monthly_ending_cash_case: np.ndarray

    # Structured tables
    line_items: tuple[LineItemRecord, ...]
    line_items_df: pd.DataFrame
    fs_month_df: pd.DataFrame     # 60 months (Jan-27 to Dec-31) 3-Statement Table
    fs_annual_df: pd.DataFrame    # 5 years (FY 2027 to FY 2031) 3-Statement Table
    cash_flow_60m_df: pd.DataFrame # Dedicated 60-Month Cash Flow & Cash Balance Table
    month_labels: tuple[str, ...] # 60 month labels
    year_labels: tuple[str, ...]  # 5 year labels


def verify_and_load_adaptive_pricing(
    cfg: FinancialModelConfig,
    override_dir: Path | None = None,
) -> tuple[AdaptivePricingIntegration, tuple[CourtRegimeMetrics, ...]]:
    """
    Ensure that AdaptivePricing/outputs/ contains all required reports and extract
    the ML Adaptive Court Hire revenue and regime utilization metrics.
    """
    ap_dir = (override_dir or cfg.resolved_adaptive_pricing_output_dir).resolve()
    if not ap_dir.exists():
        raise FileNotFoundError(
            f"AdaptivePricing outputs directory not found at {ap_dir}. "
            "Run `uv run main` from CaseComp so AdaptivePricing executes first."
        )

    missing_or_empty = []
    for req_file in REQUIRED_ADAPTIVE_PRICING_REPORTS:
        fpath = ap_dir / req_file
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing_or_empty.append(req_file)

    if missing_or_empty:
        raise RuntimeError(
            f"AdaptivePricing/outputs/ is missing required report files: {missing_or_empty} in {ap_dir}."
        )

    daily_df = pd.read_csv(ap_dir / "daily_revenue_comparison.csv")
    sched_df = pd.read_csv(ap_dir / "monthly_calendar_pricing_schedule.csv")

    sim_base_rev = float(daily_df["baseline_rev_mvnd"].sum())
    ml_rev = float(daily_df["ml_adaptive_rev_mvnd"].sum())
    case_base_rev = cfg.revenue.case_baseline_court_hire_mvnd

    lift_vs_case = ml_rev - case_base_rev
    lift_vs_case_pct = (lift_vs_case / case_base_rev) * 100.0
    lift_vs_sim = ml_rev - sim_base_rev
    lift_vs_sim_pct = (lift_vs_sim / sim_base_rev) * 100.0

    sched_df["start_hour"] = sched_df["hour_of_day"].str.slice(0, 2).astype(int)
    sched_df["is_weekend"] = sched_df["day_of_week"].isin(["Saturday", "Sunday"])

    wd_day_mask = (~sched_df["is_weekend"]) & (sched_df["start_hour"] < 17)
    wd_eve_mask = (~sched_df["is_weekend"]) & (sched_df["start_hour"] >= 17)
    we_mask = sched_df["is_weekend"]

    ops = cfg.operations

    def _regime_stats(mask: pd.Series) -> tuple[float, float, float, float]:
        sub = sched_df.loc[mask]
        courts_col = "mc_mean_ml_expected_courts" if "mc_mean_ml_expected_courts" in sub.columns else "ml_expected_courts"
        price_col = "mc_mean_ml_price_kvnd" if "mc_mean_ml_price_kvnd" in sub.columns else "ml_adaptive_price_kvnd"
        ml_booked = float(sub[courts_col].sum())
        avail = float(len(sub) * ops.courts_count)
        ml_util = ml_booked / avail if avail > 0 else 0.0
        rev_kvnd = float((sub[courts_col] * sub[price_col]).sum())
        avg_price = float(sub[price_col].mean())
        return ml_util, ml_booked, avg_price, rev_kvnd / 1000.0

    wd_day_u, wd_day_b, wd_day_p, wd_day_r = _regime_stats(wd_day_mask)
    wd_eve_u, wd_eve_b, wd_eve_p, wd_eve_r = _regime_stats(wd_eve_mask)
    we_u, we_b, we_p, we_r = _regime_stats(we_mask)

    raw_sched_rev_sum = wd_day_r + wd_eve_r + we_r
    scale_factor = ml_rev / raw_sched_rev_sum if raw_sched_rev_sum > 0 else 1.0
    wd_day_r *= scale_factor
    wd_eve_r *= scale_factor
    we_r *= scale_factor

    avail_wd_day = float(ops.courts_count * ops.weekdays_per_month * ops.weekday_daytime_hours_per_day)
    avail_wd_eve = float(ops.courts_count * ops.weekdays_per_month * ops.weekday_evening_hours_per_day)
    avail_we = float(ops.courts_count * ops.weekend_days_per_month * ops.weekend_hours_per_day)

    regimes = (
        CourtRegimeMetrics(
            regime_name="Weekday Daytime (06:00–17:00)",
            days_per_month=ops.weekdays_per_month,
            hours_per_day=ops.weekday_daytime_hours_per_day,
            available_court_hours=avail_wd_day,
            case_utilisation_rate=ops.weekday_daytime_utilisation,
            case_booked_hours=avail_wd_day * ops.weekday_daytime_utilisation,
            case_price_kvnd=ops.weekday_daytime_price_kvnd,
            case_revenue_mvnd=(avail_wd_day * ops.weekday_daytime_utilisation * ops.weekday_daytime_price_kvnd) / 1000.0,
            ml_utilisation_rate=wd_day_u,
            ml_booked_hours=wd_day_b,
            ml_avg_price_kvnd=wd_day_p,
            ml_revenue_mvnd=wd_day_r,
        ),
        CourtRegimeMetrics(
            regime_name="Weekday Evening (17:00–22:00)",
            days_per_month=ops.weekdays_per_month,
            hours_per_day=ops.weekday_evening_hours_per_day,
            available_court_hours=avail_wd_eve,
            case_utilisation_rate=ops.weekday_evening_utilisation,
            case_booked_hours=avail_wd_eve * ops.weekday_evening_utilisation,
            case_price_kvnd=ops.weekday_evening_price_kvnd,
            case_revenue_mvnd=(avail_wd_eve * ops.weekday_evening_utilisation * ops.weekday_evening_price_kvnd) / 1000.0,
            ml_utilisation_rate=wd_eve_u,
            ml_booked_hours=wd_eve_b,
            ml_avg_price_kvnd=wd_eve_p,
            ml_revenue_mvnd=wd_eve_r,
        ),
        CourtRegimeMetrics(
            regime_name="Weekend All Day (06:00–22:00)",
            days_per_month=ops.weekend_days_per_month,
            hours_per_day=ops.weekend_hours_per_day,
            available_court_hours=avail_we,
            case_utilisation_rate=ops.weekend_utilisation,
            case_booked_hours=avail_we * ops.weekend_utilisation,
            case_price_kvnd=ops.weekend_price_kvnd,
            case_revenue_mvnd=(avail_we * ops.weekend_utilisation * ops.weekend_price_kvnd) / 1000.0,
            ml_utilisation_rate=we_u,
            ml_booked_hours=we_b,
            ml_avg_price_kvnd=we_p,
            ml_revenue_mvnd=we_r,
        ),
    )

    base_courts_col = (
        "mc_mean_baseline_expected_courts"
        if "mc_mean_baseline_expected_courts" in sched_df.columns
        else "baseline_expected_courts"
    )
    ml_courts_col = (
        "mc_mean_ml_expected_courts"
        if "mc_mean_ml_expected_courts" in sched_df.columns
        else "ml_expected_courts"
    )
    sim_base_booked = float(sched_df[base_courts_col].sum())
    ml_booked_total = float(sched_df[ml_courts_col].sum())
    total_avail = avail_wd_day + avail_wd_eve + avail_we

    integration = AdaptivePricingIntegration(
        output_dir=ap_dir,
        verified_files=REQUIRED_ADAPTIVE_PRICING_REPORTS,
        sim_baseline_court_hire_mvnd=sim_base_rev,
        case_baseline_court_hire_mvnd=case_base_rev,
        ml_adaptive_court_hire_mvnd=ml_rev,
        lift_vs_case_mvnd=lift_vs_case,
        lift_vs_case_pct=lift_vs_case_pct,
        lift_vs_sim_baseline_mvnd=lift_vs_sim,
        lift_vs_sim_baseline_pct=lift_vs_sim_pct,
        sim_baseline_booked_hours=sim_base_booked,
        ml_adaptive_booked_hours=ml_booked_total,
        ml_blended_util_pct=(ml_booked_total / total_avail) * 100.0,
    )
    return integration, regimes


def verify_and_load_timeline_costs(
    cfg: FinancialModelConfig,
    override_dir: Path | None = None,
) -> TimelineCostIntegration:
    """
    Ensure that PickSocialProjectTimeline/outputs/ contains the leveled monthly cost
    allocation and recurring platform OpEx reports, and load them into FinancialModel.
    """
    tl_dir = (override_dir or cfg.resolved_timeline_output_dir).resolve()
    if not tl_dir.exists():
        raise FileNotFoundError(
            f"PickSocialProjectTimeline outputs directory not found at {tl_dir}. "
            "Run `uv run main` from CaseComp so PickSocialProjectTimeline executes before FinancialModel."
        )

    missing_or_empty = []
    for req_file in REQUIRED_TIMELINE_REPORTS:
        fpath = tl_dir / req_file
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing_or_empty.append(req_file)

    if missing_or_empty:
        raise RuntimeError(
            f"PickSocialProjectTimeline/outputs/ is missing required report files: {missing_or_empty} in {tl_dir}."
        )

    alloc_df = pd.read_csv(tl_dir / "monthly_cost_allocation.csv")
    committed_6m = alloc_df["leveled_committed_base_mvnd"].to_numpy(dtype=float)
    reserve_6m = alloc_df["leveled_contingency_reserve_mvnd"].to_numpy(dtype=float)
    max_buf_6m = alloc_df["leveled_max_buffered_total_mvnd"].to_numpy(dtype=float)

    return TimelineCostIntegration(
        output_dir=tl_dir,
        verified_files=REQUIRED_TIMELINE_REPORTS,
        monthly_committed_dev_base_6m=committed_6m,
        monthly_contingency_reserve_6m=reserve_6m,
        monthly_max_buffered_dev_6m=max_buf_6m,
        total_committed_dev_base_mvnd=float(committed_6m.sum()),
        total_contingency_reserve_mvnd=float(reserve_6m.sum()),
        total_max_buffered_dev_mvnd=float(max_buf_6m.sum()),
        launch_platform_opex_m5_m6_mvnd=cfg.platform_opex.launch_months_5_6_mean_mvnd,
        steady_state_platform_opex_m7_plus_mvnd=cfg.platform_opex.steady_state_months_7_plus_mean_mvnd,
    )


def verify_and_load_venue_scaling(
    cfg: FinancialModelConfig,
    override_dir: Path | None = None,
) -> VenueScalingIntegration:
    """
    Ensure that VenueScalingTimeline/outputs/ contains all required reports and load
    the 60-month court capacity, even 1,750M/mo CapEx split (Jul–Oct 2027), and lease schedule.
    """
    vs_dir = (override_dir or cfg.resolved_venue_scaling_output_dir).resolve()
    if not vs_dir.exists():
        raise FileNotFoundError(
            f"VenueScalingTimeline outputs directory not found at {vs_dir}. "
            "Run `uv run main` from CaseComp so VenueScalingTimeline executes before FinancialModel."
        )

    missing_or_empty = []
    for req_file in REQUIRED_VENUE_SCALING_REPORTS:
        fpath = vs_dir / req_file
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing_or_empty.append(req_file)

    if missing_or_empty:
        raise RuntimeError(
            f"VenueScalingTimeline/outputs/ is missing required report files: {missing_or_empty} in {vs_dir}."
        )

    m60_df = pd.read_csv(vs_dir / "monthly_expansion_capex_and_ramp.csv")
    active_courts = m60_df["active_courts_total"].to_numpy(dtype=int)
    capex_60m = m60_df["fitout_capex_even_split_mvnd"].to_numpy(dtype=float)
    lease_60m = m60_df["venue2_lease_mvnd"].to_numpy(dtype=float)
    fill_60m = m60_df["venue2_fill_up_factor_mean"].to_numpy(dtype=float)

    vs_cfg = cfg.venue_scaling
    return VenueScalingIntegration(
        output_dir=vs_dir,
        verified_files=REQUIRED_VENUE_SCALING_REPORTS,
        active_courts_60m=active_courts,
        fitout_capex_even_split_60m=capex_60m,
        venue2_lease_60m=lease_60m,
        venue2_fill_factor_mean_60m=fill_60m,
        total_fitout_capex_mvnd=vs_cfg.total_fitout_capex_mvnd,
        even_monthly_fitout_capex_mvnd=vs_cfg.even_monthly_fitout_capex_mvnd,
        investor_equity_injection_mvnd=vs_cfg.investor_total_new_money_mvnd,
        investor_reserve_mvnd=vs_cfg.investor_reserve_mvnd,
    )


def verify_and_load_marketing(
    cfg: FinancialModelConfig,
    override_dir: Path | None = None,
) -> MarketingIntegration:
    """
    Ensure that Marketing/outputs/ contains all required reports and load the ML-predicted
    4-segment marketing budget, social follower funnel (~8.57% conversion), segment visits,
    and the 12-distribution ancillary revenue lines (Paddle/Balls, Coaching, Food & Drinks).
    """
    mkt_dir = (override_dir or cfg.resolved_marketing_output_dir).resolve()
    if not mkt_dir.exists():
        raise FileNotFoundError(
            f"Marketing outputs directory not found at {mkt_dir}. "
            "Run `uv run main` from CaseComp so Marketing executes before FinancialModel."
        )

    missing_or_empty = []
    for req_file in REQUIRED_MARKETING_REPORTS:
        fpath = mkt_dir / req_file
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing_or_empty.append(req_file)

    if missing_or_empty:
        raise RuntimeError(
            f"Marketing/outputs/ is missing required report files: {missing_or_empty} in {mkt_dir}."
        )

    mkt_df = pd.read_csv(mkt_dir / "monthly_segment_marketing_and_visits_60m.csv")
    anc_df = pd.read_csv(mkt_dir / "segment_ancillary_revenue_60m.csv")

    return MarketingIntegration(
        output_dir=mkt_dir,
        verified_files=REQUIRED_MARKETING_REPORTS,
        followers_ml_60m=mkt_df["Social_Followers_ML"].to_numpy(dtype=float),
        follower_to_customer_conv_rate_60m=mkt_df["Follower_to_Customer_Conv_Rate"].to_numpy(dtype=float),
        customers_total_ml_60m=mkt_df["Unique_Customers_Total_ML"].to_numpy(dtype=float),
        visits_competitive_ml_60m=mkt_df["Visits_Competitive_ML"].to_numpy(dtype=float),
        visits_social_ml_60m=mkt_df["Visits_Social_ML"].to_numpy(dtype=float),
        visits_trend_explorers_ml_60m=mkt_df["Visits_Trend_Explorers_ML"].to_numpy(dtype=float),
        visits_families_ml_60m=mkt_df["Visits_Families_ML"].to_numpy(dtype=float),
        visits_total_ml_60m=mkt_df["Visits_Total_ML"].to_numpy(dtype=float),
        mkt_budget_competitive_ml_60m=mkt_df["Mkt_Budget_Competitive_ML_MVND"].to_numpy(dtype=float),
        mkt_budget_social_ml_60m=mkt_df["Mkt_Budget_Social_ML_MVND"].to_numpy(dtype=float),
        mkt_budget_trend_explorers_ml_60m=mkt_df["Mkt_Budget_Trend_Explorers_ML_MVND"].to_numpy(dtype=float),
        mkt_budget_families_ml_60m=mkt_df["Mkt_Budget_Families_ML_MVND"].to_numpy(dtype=float),
        mkt_budget_total_ml_60m=mkt_df["Mkt_Budget_Total_ML_MVND"].to_numpy(dtype=float),
        paddle_balls_total_ml_60m=anc_df["Paddle_Balls_Total_ML_MVND"].to_numpy(dtype=float),
        coaching_total_ml_60m=anc_df["Coaching_Total_ML_MVND"].to_numpy(dtype=float),
        food_drinks_total_ml_60m=anc_df["Food_Drinks_Total_ML_MVND"].to_numpy(dtype=float),
        ancillary_total_ml_60m=anc_df["Ancillary_Total_ML_MVND"].to_numpy(dtype=float),
    )


def _sample_month_conditioned(
    dist,
    rng: np.random.Generator,
    n_sims: int,
    shared_weight: float = 0.62,
) -> np.ndarray:
    """
    Sample `n_sims` draws for a given calendar month `m` combining a shared monthly
    realized environment condition (`shared_weight`) and an independent trajectory draw
    (`sqrt(1 - shared_weight^2)`), preserving 100% of `dist.mean`, `dist.std`, and `[min, max]`.
    This ensures both our Integrated Solution and the Case Baseline fluctuate month-to-month
    around their benchmark means under the exact same monthly conditions while maintaining full
    Monte Carlo cross-trajectory statistical bands.
    """
    if dist.std <= 0.0:
        return np.full(n_sims, dist.mean, dtype=float)
    z_month = float(rng.standard_normal())
    z_sims = rng.standard_normal(size=n_sims)
    idio_weight = float(np.sqrt(max(0.0, 1.0 - shared_weight ** 2)))
    raw = dist.mean + dist.std * (shared_weight * z_month + idio_weight * z_sims)
    return np.clip(raw, dist.min, dist.max)


def _run_monte_carlo_financial_simulations(
    cfg: FinancialModelConfig,
    court_hire_base_mvnd: float,
    timeline_cost: TimelineCostIntegration,
    venue_scaling: VenueScalingIntegration,
    marketing: MarketingIntegration,
    month_names: list[str],
) -> tuple[MonteCarloSubscriptionSummary, dict[str, np.ndarray]]:
    """
    Execute N=300 Monte Carlo 60-month (5-Year) simulation trajectories (with 80/20 train/test
    holdout split + -30% demand stress test + Stochastic Case Baseline under the exact same
    distributions and simulation lines), sampling from all distributions in `configs/*.toml`
    including:
      - External macro pickleball interest trend multiplier `I_t = I_{t-1} * (1 + g_t)` (`externals.toml`)
      - Legal/ward post-21:00 noise handling cost, U-18 safety compliance, weather, competitor drag, inflation (`externals.toml`)
      - Path C custom platform build (`M1–M6`) & membership subscriptions (`M5+`)
      - Path A 14-court second club expansion (`M10–M13` build @ 1,750M/mo CapEx + 330M/mo lease; `M14+` 20 courts live with 6–9 month fill-up distribution using OUR model).
    """
    rng = np.random.default_rng(cfg.randomness.random_seed)
    n_sims = cfg.randomness.n_simulations
    n_months = len(month_names)  # 60
    n_train = int(round(n_sims * cfg.randomness.train_fraction))
    n_test = n_sims - n_train

    cust_cfg = cfg.customers
    mp_cfg = cust_cfg.membership
    sv_cfg = cust_cfg.survey_exhibit3
    sm_cfg = cust_cfg.social_media
    cogs_cfg = cfg.cogs
    opex_cfg = cfg.opex
    sga_cfg = cfg.sga
    v2_opex_cfg = cfg.venue2_opex_sga
    popex_cfg = cfg.platform_opex
    vs_cfg = cfg.venue_scaling
    ext_cfg = cfg.externals
    rand_cfg = cfg.randomness
    bs_cfg = cfg.balance_sheet

    launch_m_idx = mp_cfg.launch_month_index - 1          # 0-based: index 4 = Month 5 (May-27)
    post_hc_m_idx = popex_cfg.post_hypercare_month_index - 1  # 0-based: index 6 = Month 7 (Jul-27)
    v2_build_start_idx = vs_cfg.build_start_month_index - 1   # 0-based: index 9 = Month 10 (Oct-27)
    v2_build_end_idx = vs_cfg.build_end_month_index - 1       # 0-based: index 12 = Month 13 (Jan-28)
    v2_open_idx = vs_cfg.open_month_index - 1                 # 0-based: index 13 = Month 14 (Feb-28)
    v2_lease_start_idx = vs_cfg.lease_start_month_index - 1   # 0-based: index 9 = Month 10 (Oct-27)
    inv_sign_idx = vs_cfg.investor_signing_tranche_month_index - 1  # 0-based: index 0 = Month 1 (Jan-27)
    inv_inj_idx = vs_cfg.investor_equity_injection_month_index - 1  # 0-based: index 9 = Month 10 (Oct-27)

    cap_ratio_new = vs_cfg.capacity_ratio_new_vs_existing     # 14.0 / 6.0 = 2.333333

    # Allocate (n_sims, n_months) arrays for base plan
    trend_idx_mat = np.zeros((n_sims, n_months))
    adj_visits_mat = np.zeros((n_sims, n_months))
    u_cust_mat = np.zeros((n_sims, n_months))
    followers_mat = np.zeros((n_sims, n_months))
    subs_mat = np.zeros((n_sims, n_months))
    new_subs_mat = np.zeros((n_sims, n_months))
    churn_subs_mat = np.zeros((n_sims, n_months))
    v2_fill_mat = np.zeros((n_sims, n_months))

    v1_court_rev_mat = np.zeros((n_sims, n_months))
    v2_court_rev_mat = np.zeros((n_sims, n_months))
    court_rev_mat = np.zeros((n_sims, n_months))
    mem_rev_mat = np.zeros((n_sims, n_months))
    paddle_rev_mat = np.zeros((n_sims, n_months))
    coach_rev_mat = np.zeros((n_sims, n_months))
    fnb_rev_mat = np.zeros((n_sims, n_months))
    v2_tot_rev_mat = np.zeros((n_sims, n_months))
    tot_rev_mat = np.zeros((n_sims, n_months))

    coach_cogs_mat = np.zeros((n_sims, n_months))
    fnb_cogs_mat = np.zeros((n_sims, n_months))
    pay_cons_cogs_mat = np.zeros((n_sims, n_months))
    mem_event_cogs_mat = np.zeros((n_sims, n_months))
    tot_cogs_mat = np.zeros((n_sims, n_months))

    v1_lease_mat = np.zeros((n_sims, n_months))
    v2_lease_mat = np.zeros((n_sims, n_months))
    lease_opex_mat = np.zeros((n_sims, n_months))
    util_opex_mat = np.zeros((n_sims, n_months))
    maint_opex_mat = np.zeros((n_sims, n_months))
    tot_opex_mat = np.zeros((n_sims, n_months))

    v1_staff_mat = np.zeros((n_sims, n_months))
    v2_staff_mat = np.zeros((n_sims, n_months))
    staff_sga_mat = np.zeros((n_sims, n_months))
    admin_sga_mat = np.zeros((n_sims, n_months))
    mkt_sga_mat = np.zeros((n_sims, n_months))
    plat_dev_sga_mat = np.zeros((n_sims, n_months))
    plat_rec_sga_mat = np.zeros((n_sims, n_months))
    ext_legal_safety_sga_mat = np.zeros((n_sims, n_months))
    tot_sga_mat = np.zeros((n_sims, n_months))

    tot_costs_mat = np.zeros((n_sims, n_months))
    ebitda_mat = np.zeros((n_sims, n_months))
    end_cash_mat = np.zeros((n_sims, n_months))

    # Stochastic Case Baseline (Status Quo: 6 courts, static pricing, no platform, no Venue 2)
    # under the EXACT SAME monthly distributions, random shocks, and simulation trajectories!
    base_tot_rev_mat = np.zeros((n_sims, n_months))
    base_tot_costs_mat = np.zeros((n_sims, n_months))
    base_ebitda_mat = np.zeros((n_sims, n_months))
    base_end_cash_mat = np.zeros((n_sims, n_months))

    # Stress-test (-30% lower demand) arrays
    stress_mult = rand_cfg.stress_demand_multiplier  # 0.70
    stress_subs_mat = np.zeros((n_sims, n_months))
    stress_tot_rev_mat = np.zeros((n_sims, n_months))
    stress_tot_costs_mat = np.zeros((n_sims, n_months))
    stress_ebitda_mat = np.zeros((n_sims, n_months))
    stress_end_cash_mat = np.zeros((n_sims, n_months))

    # Trajectory-level structural draws
    base_ceiling_draws = mp_cfg.addressable_member_ceiling_dist.sample(rng, size=n_sims)
    network_catchment_mult_draws = vs_cfg.network_customer_catchment_expansion_dist.sample(rng, size=n_sims)
    v2_fill_months_draws = vs_cfg.fill_up_duration_months_dist.sample(rng, size=n_sims)
    v2_init_util_draws = vs_cfg.initial_open_utilization_fraction_dist.sample(rng, size=n_sims)
    legal_v2_mult_draws = ext_cfg.legal_venue2_scaling_multiplier_dist.sample(rng, size=n_sims)

    init_followers_draws = sm_cfg.initial_followers_dist.sample(rng, size=n_sims)
    stated_intent_draws = sv_cfg.membership_stated_intent_rate_dist.sample(rng, size=n_sims)
    launch_conv_draws = mp_cfg.launch_conversion_of_stated_intent_dist.sample(rng, size=n_sims)

    curr_trend_idx = np.full(n_sims, ext_cfg.initial_trend_index, dtype=float)
    curr_cost_inflation_idx = np.ones(n_sims, dtype=float)
    curr_cash_sims = np.full(n_sims, bs_cfg.starting_cash_mvnd, dtype=float)
    curr_stress_cash_sims = np.full(n_sims, bs_cfg.starting_cash_mvnd, dtype=float)
    curr_base_cash_sims = np.full(n_sims, bs_cfg.starting_cash_mvnd, dtype=float)

    for m in range(n_months):
        # 1. Sample External Pickleball Interest Trend Rate g_t & Update Cumulative Trend Multiplier I_t
        g_t = _sample_month_conditioned(ext_cfg.monthly_interest_change_rate_dist, rng, n_sims, shared_weight=0.58)
        curr_trend_idx = np.clip(
            curr_trend_idx * (1.0 + g_t),
            ext_cfg.min_cumulative_trend_floor,
            ext_cfg.max_cumulative_trend_ceiling,
        )
        trend_idx_mat[:, m] = curr_trend_idx

        # Community retention insulation (Exhibit 3: social games & leagues buffer casual fad cooling)
        insul = ext_cfg.community_retention_insulation if m >= launch_m_idx else 0.0
        effective_demand_trend = np.where(
            curr_trend_idx < 1.0,
            1.0 - (1.0 - curr_trend_idx) * (1.0 - insul),
            curr_trend_idx,
        )

        # Sample other external factors for month m (shared across both Plan and Baseline)
        comp_drag = _sample_month_conditioned(ext_cfg.competitor_non_member_drag_rate_dist, rng, n_sims, shared_weight=0.58)
        weather_mult = _sample_month_conditioned(ext_cfg.weather_utilization_multiplier_dist, rng, n_sims, shared_weight=0.68)
        infl_rate = _sample_month_conditioned(ext_cfg.input_cost_drift_rate_dist, rng, n_sims, shared_weight=0.58)
        curr_cost_inflation_idx = curr_cost_inflation_idx * (1.0 + infl_rate)

        # 2. Venue 2 (14 Courts) Fill-Up Factor f_k (0 in Months 1..13; ramps over 6–9 months from Month 14 / Feb-28)
        if m < v2_open_idx:
            f_v2 = np.zeros(n_sims, dtype=float)
        else:
            k_open = float(m - v2_open_idx + 1)
            prog = np.clip((k_open - 1.0) / np.maximum(1.0, v2_fill_months_draws - 1.0), 0.0, 1.0)
            s_curve = 0.45 * prog + 0.55 * (3.0 * (prog ** 2) - 2.0 * (prog ** 3))
            f_v2 = np.clip(v2_init_util_draws + (1.0 - v2_init_util_draws) * s_curve, 0.0, 1.0)
        v2_fill_mat[:, m] = f_v2

        # 3. Sample Base Unique Customers & Visits from the ML Marketing Funnel (`Marketing/outputs/`),
        #    where Active Customers = Social Followers * Follower-to-Customer Conversion Rate (~8.57%)
        cust_noise = _sample_month_conditioned(cust_cfg.unique_customers_dist, rng, n_sims, shared_weight=0.65) / max(cust_cfg.unique_customers_dist.mean, 1.0)
        vis_noise = _sample_month_conditioned(cust_cfg.player_visits_anchor_dist, rng, n_sims, shared_weight=0.65) / max(cust_cfg.player_visits_anchor_dist.mean, 1.0)

        network_scale_t = 1.0 + (network_catchment_mult_draws - 1.0) * f_v2
        u_t = marketing.customers_total_ml_60m[m] * cust_noise
        u_cust_mat[:, m] = u_t

        # Total adjusted player visits across active courts driven by the 4-Segment ML Marketing Funnel
        court_capacity_mult_t = 1.0 + cap_ratio_new * f_v2
        adj_visits_mat[:, m] = marketing.visits_total_ml_60m[m] * vis_noise * weather_mult

        # Social media followers (ML Marketing Funnel trajectory with stochastic variation)
        new_fol_r = _sample_month_conditioned(sm_cfg.monthly_new_follow_rate_dist, rng, n_sims, shared_weight=0.58)
        unfol_r = _sample_month_conditioned(sm_cfg.monthly_unfollow_rate_dist, rng, n_sims, shared_weight=0.58)
        fol_noise = (init_followers_draws / max(sm_cfg.initial_followers_dist.mean, 1.0)) * (1.0 + 0.25 * (new_fol_r - unfol_r))
        followers_mat[:, m] = marketing.followers_ml_60m[m] * fol_noise

        # 4. Smooth Month-to-Month Subscriber State Evolution (Starts at Launch Month 5: May-27)
        # Dynamic addressable member ceiling expands as Venue 2 (14 courts) opens and fills up
        dynamic_ceiling_t = base_ceiling_draws * network_scale_t * effective_demand_trend

        if m < launch_m_idx:
            subs_mat[:, m] = 0.0
            new_subs_mat[:, m] = 0.0
            churn_subs_mat[:, m] = 0.0
            stress_subs_mat[:, m] = 0.0
        elif m == launch_m_idx:
            acq_shock = _sample_month_conditioned(rand_cfg.subscriber_acquisition_shock_dist, rng, n_sims, shared_weight=0.58)
            init_subs = u_t * stated_intent_draws * launch_conv_draws * acq_shock
            subs_mat[:, m] = np.clip(init_subs, 80.0, dynamic_ceiling_t)
            new_subs_mat[:, m] = subs_mat[:, m]
            churn_subs_mat[:, m] = 0.0
            stress_subs_mat[:, m] = subs_mat[:, m] * stress_mult
        else:
            prev_s = subs_mat[:, m - 1]
            churn_r = (
                _sample_month_conditioned(mp_cfg.monthly_churn_rate_dist, rng, n_sims, shared_weight=0.58)
                * _sample_month_conditioned(rand_cfg.subscriber_churn_shock_dist, rng, n_sims, shared_weight=0.58)
            )
            churn_r = np.clip(churn_r, mp_cfg.monthly_churn_rate_dist.min, mp_cfg.monthly_churn_rate_dist.max)

            new_sub_r = (
                _sample_month_conditioned(mp_cfg.monthly_new_subscriber_rate_dist, rng, n_sims, shared_weight=0.58)
                * _sample_month_conditioned(rand_cfg.subscriber_acquisition_shock_dist, rng, n_sims, shared_weight=0.58)
            )
            new_sub_r = np.clip(new_sub_r, mp_cfg.monthly_new_subscriber_rate_dist.min, mp_cfg.monthly_new_subscriber_rate_dist.max)

            churned = prev_s * churn_r
            non_sub_pool = np.maximum(0.0, u_t - prev_s)
            headroom = np.clip(1.0 - (prev_s / dynamic_ceiling_t), 0.0, 1.0)
            acquired = non_sub_pool * new_sub_r * headroom

            curr_s = np.clip(prev_s - churned + acquired, 0.0, dynamic_ceiling_t)
            subs_mat[:, m] = curr_s
            new_subs_mat[:, m] = acquired
            churn_subs_mat[:, m] = churned

            # Stress test (-30% demand) smooth subscriber path
            prev_stress_s = stress_subs_mat[:, m - 1]
            stress_u_t = u_t * stress_mult
            stress_ceiling = dynamic_ceiling_t * stress_mult
            stress_churned = prev_stress_s * churn_r
            stress_pool = np.maximum(0.0, stress_u_t - prev_stress_s)
            stress_headroom = np.clip(1.0 - (prev_stress_s / stress_ceiling), 0.0, 1.0)
            stress_acq = stress_pool * new_sub_r * stress_headroom
            stress_subs_mat[:, m] = np.clip(prev_stress_s - stress_churned + stress_acq, 0.0, stress_ceiling)

        # 5. Sample Monthly Revenue Lines (M VND) — Using OUR Model for 6 -> 20 Courts + External Multipliers
        macro_shock = _sample_month_conditioned(rand_cfg.monthly_demand_macro_shock_dist, rng, n_sims, shared_weight=0.68)
        anc_shock = _sample_month_conditioned(rand_cfg.ancillary_revenue_shock_dist, rng, n_sims, shared_weight=0.68)

        # External multiplier on court utilization (trend + weather + minor competitor drag on non-members)
        ext_court_mult = effective_demand_trend * weather_mult * (1.0 + 0.5 * comp_drag)

        v1_court_r = court_hire_base_mvnd * macro_shock * ext_court_mult
        v2_court_r = (court_hire_base_mvnd * cap_ratio_new) * f_v2 * macro_shock * ext_court_mult
        court_r = v1_court_r + v2_court_r

        mem_r = subs_mat[:, m] * mp_cfg.monthly_fee_mvnd

        # Ancillary revenues (Paddle/Balls, Coaching, Food & Drinks) driven by the 12 Segment-by-Ancillary
        # Spending Distributions in `Marketing/outputs/segment_ancillary_revenue_60m.csv`
        v2_share_t = (cap_ratio_new * f_v2) / np.maximum(1.0 + cap_ratio_new * f_v2, 1.0)
        pad_r = marketing.paddle_balls_total_ml_60m[m] * anc_shock * weather_mult
        coach_r = marketing.coaching_total_ml_60m[m] * anc_shock * weather_mult
        fnb_r = marketing.food_drinks_total_ml_60m[m] * anc_shock * weather_mult

        v2_pad_r = pad_r * v2_share_t
        v2_coach_r = coach_r * v2_share_t
        v2_fnb_r = fnb_r * v2_share_t

        v2_tot_r = v2_court_r + v2_pad_r + v2_coach_r + v2_fnb_r
        tot_r = court_r + mem_r + pad_r + coach_r + fnb_r

        v1_court_rev_mat[:, m] = v1_court_r
        v2_court_rev_mat[:, m] = v2_court_r
        court_rev_mat[:, m] = court_r
        mem_rev_mat[:, m] = mem_r
        paddle_rev_mat[:, m] = pad_r
        coach_rev_mat[:, m] = coach_r
        fnb_rev_mat[:, m] = fnb_r
        v2_tot_rev_mat[:, m] = v2_tot_r
        tot_rev_mat[:, m] = tot_r

        # 6. Sample Monthly COGS Lines (M VND)
        coach_share = _sample_month_conditioned(cogs_cfg.coach_fees_share_dist, rng, n_sims, shared_weight=0.65)
        c_coach = coach_r * coach_share

        fnb_share = _sample_month_conditioned(cogs_cfg.fnb_supplies_share_dist, rng, n_sims, shared_weight=0.65)
        c_fnb = fnb_r * fnb_share

        pay_pct = (
            _sample_month_conditioned(cogs_cfg.payment_fee_pct_of_revenue_dist, rng, n_sims, shared_weight=0.62)
            * _sample_month_conditioned(rand_cfg.payment_gateway_cash_mix_shock_dist, rng, n_sims, shared_weight=0.62)
        )
        pay_pct = np.clip(
            pay_pct,
            cogs_cfg.payment_fee_pct_of_revenue_dist.min,
            cogs_cfg.payment_fee_pct_of_revenue_dist.max,
        )
        base_consumables_c = _sample_month_conditioned(cogs_cfg.consumables_replacement_monthly_dist, rng, n_sims, shared_weight=0.62)
        consumables_c = (
            base_consumables_c
            * court_capacity_mult_t
            * curr_cost_inflation_idx
        )
        c_pay_cons = tot_r * pay_pct + consumables_c

        per_mem_event_kvnd = _sample_month_conditioned(cogs_cfg.membership_event_partner_court_per_member_kvnd_dist, rng, n_sims, shared_weight=0.62)
        # Once Venue 2 (14 courts) opens in Month 14, PICK! hosts most member leagues on its own 20 courts
        # rather than renting external partner courts @ 200k/hr, reducing external partner-court COGS per member by 45%
        own_venue_event_efficiency = 1.0 - 0.45 * f_v2
        c_mem_event = subs_mat[:, m] * (per_mem_event_kvnd / 1000.0) * own_venue_event_efficiency

        c_tot_cogs = c_coach + c_fnb + c_pay_cons + c_mem_event

        coach_cogs_mat[:, m] = c_coach
        fnb_cogs_mat[:, m] = c_fnb
        pay_cons_cogs_mat[:, m] = c_pay_cons
        mem_event_cogs_mat[:, m] = c_mem_event
        tot_cogs_mat[:, m] = c_tot_cogs

        # 7. Sample Facility OpEx Lines (Venue 1 [6 Courts] + Venue 2 [14 Courts from Q4 2027])
        o_v1_lease = _sample_month_conditioned(opex_cfg.venue_lease_dist, rng, n_sims, shared_weight=0.62)
        o_v2_lease = (
            _sample_month_conditioned(v2_opex_cfg.venue_lease_dist, rng, n_sims, shared_weight=0.62)
            if m >= v2_lease_start_idx
            else np.zeros(n_sims)
        )
        o_lease = o_v1_lease + o_v2_lease

        base_v1_util = _sample_month_conditioned(opex_cfg.electricity_lighting_water_dist, rng, n_sims, shared_weight=0.68)
        o_v1_util = base_v1_util * curr_cost_inflation_idx
        if m >= v2_open_idx:
            o_v2_util = (
                _sample_month_conditioned(v2_opex_cfg.electricity_lighting_water_dist, rng, n_sims, shared_weight=0.68)
                * (0.65 + 0.35 * f_v2)
                * curr_cost_inflation_idx
            )
            o_v2_maint = _sample_month_conditioned(v2_opex_cfg.maintenance_and_insurance_dist, rng, n_sims, shared_weight=0.68) * curr_cost_inflation_idx
        else:
            o_v2_util = np.zeros(n_sims)
            o_v2_maint = np.zeros(n_sims)

        base_v1_maint = _sample_month_conditioned(opex_cfg.maintenance_and_insurance_dist, rng, n_sims, shared_weight=0.68)
        o_v1_maint = base_v1_maint * curr_cost_inflation_idx
        o_util = o_v1_util + o_v2_util
        o_maint = o_v1_maint + o_v2_maint
        o_tot = o_lease + o_util + o_maint

        v1_lease_mat[:, m] = o_v1_lease
        v2_lease_mat[:, m] = o_v2_lease
        lease_opex_mat[:, m] = o_lease
        util_opex_mat[:, m] = o_util
        maint_opex_mat[:, m] = o_maint
        tot_opex_mat[:, m] = o_tot

        # 8. Sample SG&A + Platform Build (M1..M6) + Platform OpEx (M5+) + External Legal/Noise & U-18 Safety Costs
        s_v1_staff = _sample_month_conditioned(sga_cfg.staff_wages_dist, rng, n_sims, shared_weight=0.65)
        s_v2_staff = (
            _sample_month_conditioned(v2_opex_cfg.staff_wages_dist, rng, n_sims, shared_weight=0.65)
            if m >= v2_open_idx
            else np.zeros(n_sims)
        )
        s_staff = s_v1_staff + s_v2_staff
        s_admin = _sample_month_conditioned(sga_cfg.booking_system_and_admin_dist, rng, n_sims, shared_weight=0.65)
        base_s_mkt = _sample_month_conditioned(sga_cfg.marketing_dist, rng, n_sims, shared_weight=0.68)
        mkt_noise = base_s_mkt / max(sga_cfg.marketing_dist.mean, 1.0)
        s_mkt = marketing.mkt_budget_total_ml_60m[m] * mkt_noise

        if m < 6:
            buf_util = _sample_month_conditioned(popex_cfg.task_buffer_utilization_dist, rng, n_sims, shared_weight=0.58)
            s_plat_dev = (
                timeline_cost.monthly_committed_dev_base_6m[m]
                + timeline_cost.monthly_contingency_reserve_6m[m] * buf_util
            )
        else:
            s_plat_dev = np.zeros(n_sims)

        if m < launch_m_idx:
            s_plat_rec = np.zeros(n_sims)
        elif m < post_hc_m_idx:
            s_plat_rec = (
                _sample_month_conditioned(popex_cfg.cloud_hosting_db_redis_dist, rng, n_sims, shared_weight=0.58)
                + _sample_month_conditioned(popex_cfg.zalo_zns_otp_notifications_dist, rng, n_sims, shared_weight=0.58)
                + _sample_month_conditioned(popex_cfg.observability_cicd_security_saas_dist, rng, n_sims, shared_weight=0.58)
                + _sample_month_conditioned(popex_cfg.school_university_community_ops_dist, rng, n_sims, shared_weight=0.58)
            )
        else:
            s_plat_rec = (
                _sample_month_conditioned(popex_cfg.cloud_hosting_db_redis_dist, rng, n_sims, shared_weight=0.58)
                + _sample_month_conditioned(popex_cfg.zalo_zns_otp_notifications_dist, rng, n_sims, shared_weight=0.58)
                + _sample_month_conditioned(popex_cfg.observability_cicd_security_saas_dist, rng, n_sims, shared_weight=0.58)
                + _sample_month_conditioned(popex_cfg.school_university_community_ops_dist, rng, n_sims, shared_weight=0.58)
                + _sample_month_conditioned(popex_cfg.ongoing_maintenance_patching_retainer_dist, rng, n_sims, shared_weight=0.58)
            )

        # External Legal/Ward Noise-Complaint Handling & Under-18 Safety Compliance (`externals.toml`)
        legal_base_c = _sample_month_conditioned(ext_cfg.legal_and_noise_monthly_cost_dist, rng, n_sims, shared_weight=0.65)
        legal_scale = legal_v2_mult_draws if m >= v2_open_idx else 1.0
        u18_c = _sample_month_conditioned(ext_cfg.under18_safety_monthly_cost_dist, rng, n_sims, shared_weight=0.65) * (1.35 if m >= v2_open_idx else 1.0)
        s_ext_legal_safety = legal_base_c * legal_scale + u18_c

        s_tot_sga = s_staff + s_admin + s_mkt + s_plat_dev + s_plat_rec + s_ext_legal_safety
        tot_c = c_tot_cogs + o_tot + s_tot_sga
        ebitda_m = tot_r - tot_c

        v1_staff_mat[:, m] = s_v1_staff
        v2_staff_mat[:, m] = s_v2_staff
        staff_sga_mat[:, m] = s_staff
        admin_sga_mat[:, m] = s_admin
        mkt_sga_mat[:, m] = s_mkt
        plat_dev_sga_mat[:, m] = s_plat_dev
        plat_rec_sga_mat[:, m] = s_plat_rec
        ext_legal_safety_sga_mat[:, m] = s_ext_legal_safety
        tot_sga_mat[:, m] = s_tot_sga
        tot_costs_mat[:, m] = tot_c
        ebitda_mat[:, m] = ebitda_m

        # 8B. Stochastic Case Baseline (Status Quo: 6 courts, static 160k/260k pricing, no PICK! Social,
        #     no Venue 2 expansion, static 25M marketing distribution) evaluated under the EXACT SAME
        #     monthly demand, weather, ancillary, COGS, OpEx, SG&A, and external draws!
        #     Centered around standard given numbers (~640M Rev, ~-480M Cost, ~+160M Profit) while
        #     fluctuating month-to-month and across N=300 trajectories under identical conditions.
        base_ext_demand = (1.0 + 0.45 * (curr_trend_idx - 1.0)) * weather_mult * (1.0 + 0.35 * (comp_drag - ext_cfg.competitor_non_member_drag_rate_dist.mean))
        base_court_r = cfg.revenue.case_baseline_court_hire_mvnd * macro_shock * base_ext_demand
        base_anc_mult = vis_noise * anc_shock * weather_mult * (1.0 + 0.45 * (curr_trend_idx - 1.0))
        base_pad_r = cfg.revenue.paddle_rental_and_balls_mvnd * base_anc_mult
        base_coach_r = cfg.revenue.coaching_mvnd * base_anc_mult
        base_fnb_r = cfg.revenue.food_and_drinks_mvnd * base_anc_mult
        base_tot_r = base_court_r + base_pad_r + base_coach_r + base_fnb_r

        base_c_coach = base_coach_r * coach_share
        base_c_fnb = base_fnb_r * fnb_share
        base_c_pay_cons = base_tot_r * pay_pct + base_consumables_c * (1.0 + 0.50 * (curr_cost_inflation_idx - 1.0))
        base_tot_cogs = base_c_coach + base_c_fnb + base_c_pay_cons

        base_o_util = base_v1_util * (1.0 + 0.50 * (curr_cost_inflation_idx - 1.0))
        base_o_maint = base_v1_maint * (1.0 + 0.50 * (curr_cost_inflation_idx - 1.0))
        base_tot_opex = o_v1_lease + base_o_util + base_o_maint

        base_legal_centered = legal_base_c - ext_cfg.legal_and_noise_monthly_cost_dist.mean
        base_tot_sga = s_v1_staff + s_admin + base_s_mkt + base_legal_centered
        base_tot_c = base_tot_cogs + base_tot_opex + base_tot_sga
        base_ebitda_m = base_tot_r - base_tot_c

        base_tot_rev_mat[:, m] = base_tot_r
        base_tot_costs_mat[:, m] = base_tot_c
        base_ebitda_mat[:, m] = base_ebitda_m

        # 9. Stress Test (-30% Demand) Revenue, COGS & Profit Calculation
        st_court_r = court_r * stress_mult
        st_mem_r = stress_subs_mat[:, m] * mp_cfg.monthly_fee_mvnd
        st_pad_r = pad_r * stress_mult
        st_coach_r = coach_r * stress_mult
        st_fnb_r = fnb_r * stress_mult
        st_tot_r = st_court_r + st_mem_r + st_pad_r + st_coach_r + st_fnb_r

        st_c_coach = st_coach_r * coach_share
        st_c_fnb = st_fnb_r * fnb_share
        st_c_pay_cons = st_tot_r * pay_pct + consumables_c * 0.85
        st_c_mem_event = stress_subs_mat[:, m] * (per_mem_event_kvnd / 1000.0) * own_venue_event_efficiency
        st_tot_cogs = st_c_coach + st_c_fnb + st_c_pay_cons + st_c_mem_event
        st_tot_c = st_tot_cogs + o_tot + s_tot_sga
        st_ebitda_m = st_tot_r - st_tot_c

        stress_tot_rev_mat[:, m] = st_tot_r
        stress_tot_costs_mat[:, m] = st_tot_c
        stress_ebitda_mat[:, m] = st_ebitda_m

        # 10. Monthly Cash Balance Evolution Across Each Monte Carlo Trajectory
        #     CapEx outflow in Months 10..13 (Oct-27..Jan-28) = -1,750.0M VND/month (total -7,000.0M VND)
        #     Investor Minh Tran equity funding:
        #       - Tranche 1 at Month 1 (Jan-27 upon signing) = +500.0M VND (working capital reserve)
        #       - Tranche 2 at Month 10 (Oct-27 Q4 2027 kickoff) = +7,500.0M VND (7,000M fit-out + 500M reserve)
        capex_out_m = vs_cfg.even_monthly_fitout_capex_mvnd if (v2_build_start_idx <= m <= v2_build_end_idx) else 0.0
        if m == inv_sign_idx:
            eq_in_m = vs_cfg.investor_signing_tranche_mvnd
        elif m == inv_inj_idx:
            eq_in_m = vs_cfg.investor_expansion_tranche_mvnd
        else:
            eq_in_m = 0.0

        curr_cash_sims = curr_cash_sims + ebitda_m - capex_out_m + eq_in_m
        curr_stress_cash_sims = curr_stress_cash_sims + st_ebitda_m - capex_out_m + eq_in_m
        curr_base_cash_sims = curr_base_cash_sims + base_ebitda_m
        end_cash_mat[:, m] = curr_cash_sims
        stress_end_cash_mat[:, m] = curr_stress_cash_sims
        base_end_cash_mat[:, m] = curr_base_cash_sims

    p_low = rand_cfg.ci_low_percentile
    p_high = rand_cfg.ci_high_percentile

    # Annual (5-year: FY 2027 .. FY 2031) simulation matrices for both Model, Baseline, and Stress
    ann_rev_sims = np.array([tot_rev_mat[:, (y * 12):((y + 1) * 12)].sum(axis=1) for y in range(5)])
    ann_costs_sims = np.array([tot_costs_mat[:, (y * 12):((y + 1) * 12)].sum(axis=1) for y in range(5)])
    ann_ebitda_sims = np.array([ebitda_mat[:, (y * 12):((y + 1) * 12)].sum(axis=1) for y in range(5)])
    stress_ann_ebitda_sims = np.array([stress_ebitda_mat[:, (y * 12):((y + 1) * 12)].sum(axis=1) for y in range(5)])

    base_ann_rev_sims = np.array([base_tot_rev_mat[:, (y * 12):((y + 1) * 12)].sum(axis=1) for y in range(5)])
    base_ann_costs_sims = np.array([base_tot_costs_mat[:, (y * 12):((y + 1) * 12)].sum(axis=1) for y in range(5)])
    base_ann_ebitda_sims = np.array([base_ebitda_mat[:, (y * 12):((y + 1) * 12)].sum(axis=1) for y in range(5)])

    # Year 3 (Months 24..35, index 24:36 -> FY 2029) investor target evaluation (Minh Tran >= 3,500M VND/yr)
    y3_ebitda_sims = ebitda_mat[:, 24:36].sum(axis=1)
    y3_stress_ebitda_sims = stress_ebitda_mat[:, 24:36].sum(axis=1)
    prob_exceed_3500 = float(np.mean(y3_ebitda_sims >= 3500.0) * 100.0)

    # Build 60-month Subscriber, External Trend, Venue Scaling & Cash Cohort DataFrame for CSV export
    cohort_rows = []
    for m in range(n_months):
        cohort_rows.append(
            {
                "month_index": m + 1,
                "month_label": month_names[m],
                "year_num": (m // 12) + 1,
                "active_courts": int(venue_scaling.active_courts_60m[m]),
                "trend_multiplier_mean": round(float(trend_idx_mat[:, m].mean()), 4),
                "trend_multiplier_p05": round(float(np.percentile(trend_idx_mat[:, m], p_low)), 4),
                "trend_multiplier_p95": round(float(np.percentile(trend_idx_mat[:, m], p_high)), 4),
                "adjusted_visits_mean": round(float(adj_visits_mat[:, m].mean()), 1),
                "unique_customers_mean": round(float(u_cust_mat[:, m].mean()), 2),
                "unique_customers_p05": round(float(np.percentile(u_cust_mat[:, m], p_low)), 2),
                "unique_customers_p95": round(float(np.percentile(u_cust_mat[:, m], p_high)), 2),
                "social_followers_mean": round(float(followers_mat[:, m].mean()), 1),
                "venue2_fill_up_factor_mean": round(float(v2_fill_mat[:, m].mean()), 4),
                "active_subscribers_mean": round(float(subs_mat[:, m].mean()), 2),
                "active_subscribers_p05": round(float(np.percentile(subs_mat[:, m], p_low)), 2),
                "active_subscribers_p95": round(float(np.percentile(subs_mat[:, m], p_high)), 2),
                "active_subscribers_train_mean": round(float(subs_mat[:n_train, m].mean()), 2),
                "active_subscribers_test_mean": round(float(subs_mat[n_train:, m].mean()), 2),
                "new_subscribers_mean": round(float(new_subs_mat[:, m].mean()), 2),
                "churned_subscribers_mean": round(float(churn_subs_mat[:, m].mean()), 2),
                "membership_rev_mean_mvnd": round(float(mem_rev_mat[:, m].mean()), 2),
                "venue2_total_rev_mean_mvnd": round(float(v2_tot_rev_mat[:, m].mean()), 2),
                "total_rev_mean_mvnd": round(float(tot_rev_mat[:, m].mean()), 2),
                "total_rev_p05_mvnd": round(float(np.percentile(tot_rev_mat[:, m], p_low)), 2),
                "total_rev_p95_mvnd": round(float(np.percentile(tot_rev_mat[:, m], p_high)), 2),
                "total_costs_mean_mvnd": round(float(tot_costs_mat[:, m].mean()), 2),
                "ebitda_mean_mvnd": round(float(ebitda_mat[:, m].mean()), 2),
                "ebitda_p05_mvnd": round(float(np.percentile(ebitda_mat[:, m], p_low)), 2),
                "ebitda_p95_mvnd": round(float(np.percentile(ebitda_mat[:, m], p_high)), 2),
                "ending_cash_mean_mvnd": round(float(end_cash_mat[:, m].mean()), 2),
                "ending_cash_p05_mvnd": round(float(np.percentile(end_cash_mat[:, m], p_low)), 2),
                "ending_cash_p95_mvnd": round(float(np.percentile(end_cash_mat[:, m], p_high)), 2),
                "base_total_rev_mean_mvnd": round(float(base_tot_rev_mat[:, m].mean()), 2),
                "base_total_rev_p05_mvnd": round(float(np.percentile(base_tot_rev_mat[:, m], p_low)), 2),
                "base_total_rev_p95_mvnd": round(float(np.percentile(base_tot_rev_mat[:, m], p_high)), 2),
                "base_total_costs_mean_mvnd": round(float(base_tot_costs_mat[:, m].mean()), 2),
                "base_ebitda_mean_mvnd": round(float(base_ebitda_mat[:, m].mean()), 2),
                "base_ebitda_p05_mvnd": round(float(np.percentile(base_ebitda_mat[:, m], p_low)), 2),
                "base_ebitda_p95_mvnd": round(float(np.percentile(base_ebitda_mat[:, m], p_high)), 2),
                "base_ending_cash_mean_mvnd": round(float(base_end_cash_mat[:, m].mean()), 2),
                "base_ending_cash_p05_mvnd": round(float(np.percentile(base_end_cash_mat[:, m], p_low)), 2),
                "base_ending_cash_p95_mvnd": round(float(np.percentile(base_end_cash_mat[:, m], p_high)), 2),
                "stress_30pct_subscribers_mean": round(float(stress_subs_mat[:, m].mean()), 2),
                "stress_30pct_total_rev_mvnd": round(float(stress_tot_rev_mat[:, m].mean()), 2),
                "stress_30pct_ebitda_mvnd": round(float(stress_ebitda_mat[:, m].mean()), 2),
                "stress_30pct_ending_cash_mvnd": round(float(stress_end_cash_mat[:, m].mean()), 2),
            }
        )
    cohort_df = pd.DataFrame(cohort_rows)

    mc_summary = MonteCarloSubscriptionSummary(
        n_simulations=n_sims,
        n_train=n_train,
        n_test=n_test,
        trend_index_mean=trend_idx_mat.mean(axis=0),
        trend_index_p05=np.percentile(trend_idx_mat, p_low, axis=0),
        trend_index_p95=np.percentile(trend_idx_mat, p_high, axis=0),
        adjusted_visits_mean=adj_visits_mat.mean(axis=0),
        adjusted_visits_p05=np.percentile(adj_visits_mat, p_low, axis=0),
        adjusted_visits_p95=np.percentile(adj_visits_mat, p_high, axis=0),
        unique_customers_mean=u_cust_mat.mean(axis=0),
        unique_customers_p05=np.percentile(u_cust_mat, p_low, axis=0),
        unique_customers_p95=np.percentile(u_cust_mat, p_high, axis=0),
        social_followers_mean=followers_mat.mean(axis=0),
        active_subscribers_mean=subs_mat.mean(axis=0),
        active_subscribers_p05=np.percentile(subs_mat, p_low, axis=0),
        active_subscribers_p95=np.percentile(subs_mat, p_high, axis=0),
        Active_subscribers_train_mean=subs_mat[:n_train].mean(axis=0),
        active_subscribers_test_mean=subs_mat[n_train:].mean(axis=0),
        new_subscribers_mean=new_subs_mat.mean(axis=0),
        churned_subscribers_mean=churn_subs_mat.mean(axis=0),
        venue2_fill_factor_mean=v2_fill_mat.mean(axis=0),
        venue2_fill_factor_p05=np.percentile(v2_fill_mat, p_low, axis=0),
        venue2_fill_factor_p95=np.percentile(v2_fill_mat, p_high, axis=0),
        venue2_total_rev_mean=v2_tot_rev_mat.mean(axis=0),
        membership_rev_mean=mem_rev_mat.mean(axis=0),
        membership_rev_p05=np.percentile(mem_rev_mat, p_low, axis=0),
        membership_rev_p95=np.percentile(mem_rev_mat, p_high, axis=0),
        total_rev_mean=tot_rev_mat.mean(axis=0),
        total_rev_p05=np.percentile(tot_rev_mat, p_low, axis=0),
        total_rev_p95=np.percentile(tot_rev_mat, p_high, axis=0),
        total_costs_mean=tot_costs_mat.mean(axis=0),
        total_costs_p05=np.percentile(tot_costs_mat, p_low, axis=0),
        total_costs_p95=np.percentile(tot_costs_mat, p_high, axis=0),
        ebitda_mean=ebitda_mat.mean(axis=0),
        ebitda_p05=np.percentile(ebitda_mat, p_low, axis=0),
        ebitda_p95=np.percentile(ebitda_mat, p_high, axis=0),
        ebitda_train_mean=ebitda_mat[:n_train].mean(axis=0),
        ebitda_test_mean=ebitda_mat[n_train:].mean(axis=0),
        ending_cash_mean=end_cash_mat.mean(axis=0),
        ending_cash_p05=np.percentile(end_cash_mat, p_low, axis=0),
        ending_cash_p95=np.percentile(end_cash_mat, p_high, axis=0),
        stress_ending_cash_mean=stress_end_cash_mat.mean(axis=0),
        base_total_rev_mean=base_tot_rev_mat.mean(axis=0),
        base_total_rev_p05=np.percentile(base_tot_rev_mat, p_low, axis=0),
        base_total_rev_p95=np.percentile(base_tot_rev_mat, p_high, axis=0),
        base_total_costs_mean=base_tot_costs_mat.mean(axis=0),
        base_total_costs_p05=np.percentile(base_tot_costs_mat, p_low, axis=0),
        base_total_costs_p95=np.percentile(base_tot_costs_mat, p_high, axis=0),
        base_ebitda_mean=base_ebitda_mat.mean(axis=0),
        base_ebitda_p05=np.percentile(base_ebitda_mat, p_low, axis=0),
        base_ebitda_p95=np.percentile(base_ebitda_mat, p_high, axis=0),
        base_ending_cash_mean=base_end_cash_mat.mean(axis=0),
        base_ending_cash_p05=np.percentile(base_end_cash_mat, p_low, axis=0),
        base_ending_cash_p95=np.percentile(base_end_cash_mat, p_high, axis=0),
        base_annual_rev_mean=base_ann_rev_sims.mean(axis=1),
        base_annual_rev_p05=np.percentile(base_ann_rev_sims, p_low, axis=1),
        base_annual_rev_p95=np.percentile(base_ann_rev_sims, p_high, axis=1),
        base_annual_costs_mean=base_ann_costs_sims.mean(axis=1),
        base_annual_costs_p05=np.percentile(base_ann_costs_sims, p_low, axis=1),
        base_annual_costs_p95=np.percentile(base_ann_costs_sims, p_high, axis=1),
        base_annual_ebitda_mean=base_ann_ebitda_sims.mean(axis=1),
        base_annual_ebitda_p05=np.percentile(base_ann_ebitda_sims, p_low, axis=1),
        base_annual_ebitda_p95=np.percentile(base_ann_ebitda_sims, p_high, axis=1),
        annual_rev_p05=np.percentile(ann_rev_sims, p_low, axis=1),
        annual_rev_p95=np.percentile(ann_rev_sims, p_high, axis=1),
        annual_costs_p05=np.percentile(ann_costs_sims, p_low, axis=1),
        annual_costs_p95=np.percentile(ann_costs_sims, p_high, axis=1),
        annual_ebitda_p05=np.percentile(ann_ebitda_sims, p_low, axis=1),
        annual_ebitda_p95=np.percentile(ann_ebitda_sims, p_high, axis=1),
        stress_annual_ebitda_p05=np.percentile(stress_ann_ebitda_sims, p_low, axis=1),
        stress_annual_ebitda_p95=np.percentile(stress_ann_ebitda_sims, p_high, axis=1),
        stress_active_subscribers_mean=stress_subs_mat.mean(axis=0),
        stress_total_rev_mean=stress_tot_rev_mat.mean(axis=0),
        stress_total_costs_mean=stress_tot_costs_mat.mean(axis=0),
        stress_ebitda_mean=stress_ebitda_mat.mean(axis=0),
        sample_subscribers_paths=subs_mat[:40].copy(),
        sample_total_rev_paths=tot_rev_mat[:40].copy(),
        sample_ebitda_paths=ebitda_mat[:40].copy(),
        sample_cash_paths=end_cash_mat[:40].copy(),
        sample_trend_paths=trend_idx_mat[:40].copy(),
        sample_base_rev_paths=base_tot_rev_mat[:40].copy(),
        sample_base_ebitda_paths=base_ebitda_mat[:40].copy(),
        sample_base_cash_paths=base_end_cash_mat[:40].copy(),
        m5_launch_subscribers=float(subs_mat[:, 4].mean()),
        m12_end_y1_subscribers=float(subs_mat[:, 11].mean()),
        m24_end_y2_subscribers=float(subs_mat[:, 23].mean()),
        m36_end_y3_subscribers=float(subs_mat[:, 35].mean()),
        m60_end_y5_subscribers=float(subs_mat[:, 59].mean()),
        m24_end_y2_membership_rev_mvnd=float(mem_rev_mat[:, 23].mean()),
        y3_annual_ebitda_mean_mvnd=float(y3_ebitda_sims.mean()),
        y3_annual_ebitda_p05_mvnd=float(np.percentile(y3_ebitda_sims, p_low)),
        y3_annual_ebitda_p95_mvnd=float(np.percentile(y3_ebitda_sims, p_high)),
        y3_annual_ebitda_stress_mvnd=float(y3_stress_ebitda_sims.mean()),
        prob_exceed_investor_3500m_y3_pct=prob_exceed_3500,
        subscriber_cohort_df=cohort_df,
    )

    mean_series = {
        "v1_court_hire": v1_court_rev_mat.mean(axis=0),
        "v2_court_hire": v2_court_rev_mat.mean(axis=0),
        "court_hire": court_rev_mat.mean(axis=0),
        "membership_rev": mem_rev_mat.mean(axis=0),
        "paddle_balls": paddle_rev_mat.mean(axis=0),
        "coaching": coach_rev_mat.mean(axis=0),
        "food_drinks": fnb_rev_mat.mean(axis=0),
        "coach_fees": coach_cogs_mat.mean(axis=0),
        "fnb_supplies": fnb_cogs_mat.mean(axis=0),
        "payment_consumables": pay_cons_cogs_mat.mean(axis=0),
        "membership_event_cogs": mem_event_cogs_mat.mean(axis=0),
        "v1_lease": v1_lease_mat.mean(axis=0),
        "v2_lease": v2_lease_mat.mean(axis=0),
        "venue_lease": lease_opex_mat.mean(axis=0),
        "utilities": util_opex_mat.mean(axis=0),
        "maint_ins": maint_opex_mat.mean(axis=0),
        "v1_staff_wages": v1_staff_mat.mean(axis=0),
        "v2_staff_wages": v2_staff_mat.mean(axis=0),
        "staff_wages": staff_sga_mat.mean(axis=0),
        "booking_admin": admin_sga_mat.mean(axis=0),
        "marketing": mkt_sga_mat.mean(axis=0),
        "platform_dev_alloc": plat_dev_sga_mat.mean(axis=0),
        "platform_recurring_opex": plat_rec_sga_mat.mean(axis=0),
        "external_legal_safety_sga": ext_legal_safety_sga_mat.mean(axis=0),
    }

    return mc_summary, mean_series


def build_financial_model(
    cfg: FinancialModelConfig,
    adaptive_output_dir: Path | None = None,
    timeline_output_dir: Path | None = None,
    venue_scaling_output_dir: Path | None = None,
    marketing_output_dir: Path | None = None,
) -> FinancialModelResults:
    """
    Build the 5-year (60-month: Jan-27 to Dec-31) 3-Statement financial model integrating:
      1. AdaptivePricing ML court hire revenue (`AdaptivePricing/outputs/`),
      2. PickSocialProjectTimeline 6-month leveled dev cost allocation & recurring platform OpEx (`PickSocialProjectTimeline/outputs/`),
      3. VenueScalingTimeline Q3 2027 14-court expansion (`6` to `20` courts, even `1,750M/mo` CapEx in `M7–M10`, `330M/mo` lease, `6–9` month fill-up ramp in `M11+`),
      4. Marketing 4-Segment ML Budget Predictor, Follower-to-Customer (~8.57%) Funnel & 12-Distribution Ancillary Revenue Engine (`Marketing/outputs/`),
      5. External macro factors (`externals.toml`: monthly pickleball interest trend multiplier `I_t`, legal/ward noise handling, U-18 safety compliance, weather, inflation).
    """
    adaptive, regimes = verify_and_load_adaptive_pricing(cfg, adaptive_output_dir)
    timeline_cost = verify_and_load_timeline_costs(cfg, timeline_output_dir)
    venue_scaling = verify_and_load_venue_scaling(cfg, venue_scaling_output_dir)
    marketing = verify_and_load_marketing(cfg, marketing_output_dir)

    total_avail_hrs = sum(r.available_court_hours for r in regimes)
    total_booked_hrs = adaptive.ml_adaptive_booked_hours
    blended_util = total_booked_hrs / total_avail_hrs
    court_hire_base = (
        adaptive.ml_adaptive_court_hire_mvnd
        if cfg.company.use_adaptive_pricing_output
        else cfg.revenue.case_baseline_court_hire_mvnd
    )
    blended_rate_kvnd = (court_hire_base * 1000.0) / total_booked_hrs

    # Customer segment metrics
    exact_visits = 0.0
    total_seg_spend_kvnd = 0.0
    for seg in cfg.customers.segments:
        seg_cust = cfg.customers.unique_customers_per_month * seg.share
        seg_visits = seg_cust * seg.visits_per_month
        exact_visits += seg_visits
        total_seg_spend_kvnd += seg_visits * seg.spend_per_visit_kvnd
    weighted_spend_kvnd = total_seg_spend_kvnd / exact_visits

    # Case Static Baseline Headline Totals (Monthly M VND)
    case_court_hire = cfg.revenue.case_baseline_court_hire_mvnd
    paddle_balls = cfg.revenue.paddle_rental_and_balls_mvnd
    coaching = cfg.revenue.coaching_mvnd
    food_drinks = cfg.revenue.food_and_drinks_mvnd
    case_total_rev = case_court_hire + paddle_balls + coaching + food_drinks

    case_coach_fees = cfg.cogs.coach_fees_mvnd
    case_fb_supplies = cfg.cogs.food_and_drink_supplies_mvnd
    case_pay_consumables = cfg.cogs.payment_fees_and_consumables_mvnd
    case_total_cogs = case_coach_fees + case_fb_supplies + case_pay_consumables
    case_gross_profit = case_total_rev - case_total_cogs

    case_venue_lease = cfg.opex.venue_lease_mvnd
    case_utilities = cfg.opex.electricity_lighting_water_mvnd
    case_maint_ins = cfg.opex.maintenance_and_insurance_mvnd
    case_total_opex = case_venue_lease + case_utilities + case_maint_ins

    case_staff_wages = cfg.sga.staff_wages_mvnd
    case_booking_admin = cfg.sga.booking_system_and_admin_mvnd
    case_marketing = cfg.sga.marketing_mvnd
    case_total_sga = case_staff_wages + case_booking_admin + case_marketing
    case_total_costs = case_total_cogs + case_total_opex + case_total_sga
    case_ebitda = case_total_rev - case_total_costs

    da = cfg.taxes_and_other.depreciation_amortization_mvnd
    interest = cfg.taxes_and_other.interest_expense_mvnd
    case_ebt = case_ebitda - da - interest
    case_taxes = max(0.0, case_ebt * cfg.taxes_and_other.corporate_tax_rate)
    case_net_income = case_ebt - case_taxes

    # Calendar month labels (60 months: Jan-27 to Dec-31)
    cal_months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    years_short = ("27", "28", "29", "30", "31")
    month_names: list[str] = [f"{cm}-{ys}" for ys in years_short for cm in cal_months]
    n_months = len(month_names)  # 60

    # Run Monte Carlo Simulations across all 60 months
    mc, ms = _run_monte_carlo_financial_simulations(
        cfg, court_hire_base, timeline_cost, venue_scaling, marketing, month_names
    )

    # Extract reconciled 60-month mean series for the 3-statement tables
    m_v1_court = ms["v1_court_hire"]
    m_v2_court = ms["v2_court_hire"]
    m_court = ms["court_hire"]
    m_mem_rev = ms["membership_rev"]
    m_paddle = ms["paddle_balls"]
    m_coach_rev = ms["coaching"]
    m_fnb_rev = ms["food_drinks"]
    m_tot_rev = m_court + m_mem_rev + m_paddle + m_coach_rev + m_fnb_rev

    m_coach_cogs = ms["coach_fees"]
    m_fnb_cogs = ms["fnb_supplies"]
    m_pay_cons = ms["payment_consumables"]
    m_mem_event = ms["membership_event_cogs"]
    m_tot_cogs = m_coach_cogs + m_fnb_cogs + m_pay_cons + m_mem_event

    m_gp = m_tot_rev - m_tot_cogs
    m_gp_pct = (m_gp / m_tot_rev) * 100.0

    m_v1_lease = ms["v1_lease"]
    m_v2_lease = ms["v2_lease"]
    m_lease = ms["venue_lease"]
    m_util = ms["utilities"]
    m_maint = ms["maint_ins"]
    m_tot_opex = m_lease + m_util + m_maint

    m_v1_staff = ms["v1_staff_wages"]
    m_v2_staff = ms["v2_staff_wages"]
    m_staff = ms["staff_wages"]
    m_admin = ms["booking_admin"]
    m_mkt = ms["marketing"]
    m_plat_dev = ms["platform_dev_alloc"]
    m_plat_rec = ms["platform_recurring_opex"]
    m_ext_leg_saf = ms["external_legal_safety_sga"]
    m_tot_sga = m_staff + m_admin + m_mkt + m_plat_dev + m_plat_rec + m_ext_leg_saf

    m_tot_opex_sga = m_tot_opex + m_tot_sga
    m_tot_costs = m_tot_cogs + m_tot_opex_sga
    m_ebitda = m_tot_rev - m_tot_costs
    m_ebitda_pct = (m_ebitda / m_tot_rev) * 100.0

    m_da = np.full(n_months, da)
    m_int = np.full(n_months, interest)
    m_ebt = m_ebitda - m_da - m_int
    m_taxes = np.maximum(0.0, m_ebt * cfg.taxes_and_other.corporate_tax_rate)
    m_net_inc = m_ebt - m_taxes

    tot_5y_rev = float(m_tot_rev.sum())

    # Build Line Items Table (for Sheet 1 Breakdown)
    def _rec(
        sec: str,
        sheet: str,
        cat: str,
        item: str,
        beh: str,
        case_m: float,
        arr: np.ndarray,
        notes: str,
    ) -> LineItemRecord:
        tot5 = float(arr.sum())
        return LineItemRecord(
            section=sec,
            template_sheet=sheet,
            template_category=cat,
            line_item=item,
            behavior=beh,
            case_baseline_monthly_mvnd=case_m,
            m1_build_monthly_mvnd=float(arr[0]),
            m5_launch_monthly_mvnd=float(arr[4]),
            m11_open20_monthly_mvnd=float(arr[13]),
            m24_end_y2_monthly_mvnd=float(arr[23]),
            five_year_avg_monthly_mvnd=float(arr.mean()),
            five_year_total_mvnd=tot5,
            pct_of_5y_revenue=(tot5 / tot_5y_rev) * 100.0,
            notes=notes,
        )

    line_items = (
        _rec(
            "REVENUE", "Revenue", "Revenue Line 1A (Venue 1)",
            "Court hire: Venue 1 (6 Courts, AdaptivePricing ML)", "Revenue (Dynamic)",
            case_court_hire, m_v1_court,
            "6 existing courts optimized via AdaptivePricing + scaled by monthly macro trend I_t",
        ),
        _rec(
            "REVENUE", "Revenue", "Revenue Line 1B (Venue 2 New)",
            "Court hire: Venue 2 (14 Courts, Opens M14 / Feb-28)", "Revenue (Path A Ramp)",
            0.0, m_v2_court,
            "14 new courts opening Feb-28 after 4M build (Oct 2027 – Jan 2028), ramping over 6–9 months via OUR model",
        ),
        _rec(
            "REVENUE", "Revenue", "Revenue Line 2 (New Path C)",
            "Membership subscriptions (PICK! Social @ 399k/mo)", "Revenue (Recurring M5+)",
            0.0, m_mem_rev,
            f"Starts M5 (May-27); expands across 20-court network to {mc.m24_end_y2_subscribers:,.0f} members by End Y2",
        ),
        _rec(
            "REVENUE", "Revenue", "Revenue Line 3",
            "Paddle rental and balls (Scales 6 -> 20 Courts)", "Revenue (Ancillary)",
            paddle_balls, m_paddle,
            "Equipment rentals & ball sales across Venue 1 (6 courts) + Venue 2 (14 courts from Feb-28)",
        ),
        _rec(
            "REVENUE", "Revenue", "Revenue Line 4",
            "Coaching (Scales 6 -> 20 Courts + Coaching Studio)", "Revenue (Services)",
            coaching, m_coach_rev,
            "Individual, group & junior coaching across Venue 1 + Venue 2 coaching facilities",
        ),
        _rec(
            "REVENUE", "Revenue", "Revenue Line 5",
            "Food and drinks (Scales 6 -> 20 Courts + Specialty Cafe)", "Revenue (F&B)",
            food_drinks, m_fnb_rev,
            "Courtside beverages & specialty cafe sales across Venue 1 + Venue 2",
        ),
        _rec(
            "SUMMARY", "Revenue", "TOTAL REVENUE",
            "Total Projected Revenue (20-Court Network + Memberships)", "Subtotal",
            case_total_rev, m_tot_rev,
            f"Grows from {m_tot_rev[0]:.2f}M (M1) to {m_tot_rev[13]:,.2f}M (M14 Open) and {m_tot_rev[23]:,.2f}M (End Y2)",
        ),

        _rec(
            "COGS", "COGS", "Direct Service Costs",
            "Coach fees (paid per lesson, ~52.17% of Coaching)", "Variable (COGS)",
            case_coach_fees, m_coach_cogs,
            "52.17% of Coaching revenue across both clubs",
        ),
        _rec(
            "COGS", "COGS", "Sold Inventory",
            "Food and drink supplies (~40.0% of F&B Revenue)", "Variable (COGS)",
            case_fb_supplies, m_fnb_cogs,
            "40.00% of Food & drinks revenue across both clubs",
        ),
        _rec(
            "COGS", "COGS", "Payment & Consumables",
            "Payment gateway fees (~1.50% of Rev) & court consumables", "Variable (COGS)",
            case_pay_consumables, m_pay_cons,
            "1.50% payment gateway fee on all revenue + ball/grip consumables across active courts",
        ),
        _rec(
            "COGS", "COGS", "Partner Courts & Events (New)",
            "Membership partner courts (@ 200k/hr) & league events", "Variable (COGS M5+)",
            0.0, m_mem_event,
            "Partner-court hire & league events (45% internalized once Venue 2's 14 courts open in M14)",
        ),
        _rec(
            "SUMMARY", "COGS", "TOTAL COGS",
            "Total Cost of Goods Sold (Direct Variable)", "Subtotal",
            case_total_cogs, m_tot_cogs,
            "Direct variable cost of coaching, F&B, payment processing, consumables & member events",
        ),
        _rec(
            "SUMMARY", "FS-Month", "GROSS PROFIT",
            f"Gross Profit (5Y Gross Margin = {(m_gp.sum()/tot_5y_rev)*100:.2f}%)", "Profit Line",
            case_gross_profit, m_gp,
            "Total Revenue minus Total Direct Variable COGS",
        ),

        _rec(
            "OPEX", "SG&A / OpEx", "Rent — Venue 1 (6 Courts)",
            "Venue 1 lease (Existing 6-Court Club)", "Fixed (OpEx)",
            case_venue_lease, m_v1_lease,
            "Fixed VND 150.0M/month lease for Venue 1 (Case Exhibit 2)",
        ),
        _rec(
            "OPEX", "SG&A / OpEx", "Rent — Venue 2 (14 Courts New)",
            "Venue 2 lease (New 14-Court Club, Starts M10 / Oct-27)", "Fixed (OpEx M10+)",
            0.0, m_v2_lease,
            "Fixed VND 330.0M/month 5-year lease starting Q4 2027 (Case Page 3 Path A)",
        ),
        _rec(
            "OPEX", "SG&A / OpEx", "Utilities (6 -> 20 Courts)",
            "Electricity, lighting and water (Venue 1 + Venue 2 M14+)", "Mostly Fixed (OpEx)",
            case_utilities, m_util,
            "40M/mo Venue 1 + ~82M/mo Venue 2 (from M14) + commercial utility inflation drift",
        ),
        _rec(
            "OPEX", "SG&A / OpEx", "Insurance & Upkeep (6 -> 20 Courts)",
            "Maintenance and insurance (Venue 1 + Venue 2 M14+)", "Fixed (OpEx)",
            case_maint_ins, m_maint,
            "20M/mo Venue 1 + 38M/mo Venue 2 (from M14) + upkeep inflation drift",
        ),
        _rec(
            "SUMMARY", "SG&A / OpEx", "SUBTOTAL OpEx",
            "Subtotal Facility Operating Expenses (OpEx)", "Subtotal",
            case_total_opex, m_tot_opex,
            "210M/mo (M1–M9) -> 540M/mo (M10–M13 Build) -> ~665M/mo (M14+ 20-Court Operations)",
        ),

        _rec(
            "SGA", "SG&A", "Staff & Payroll (6 -> 20 Courts)",
            "Staff wages (Venue 1 130M + Venue 2 215M from M14)", "Fixed (SG&A)",
            case_staff_wages, m_staff,
            "130M/mo Venue 1 crew + 215M/mo Venue 2 site manager, front desk, baristas & cleaning (from Feb-28)",
        ),
        _rec(
            "SGA", "SG&A", "Tech Support & Admin",
            "Booking system and administration (Base)", "Fixed (SG&A)",
            case_booking_admin, m_admin,
            "Baseline administrative & office overhead (10.0M VND/month)",
        ),
        _rec(
            "SGA", "SG&A", "Marketing & Growth",
            "Marketing (Social & Community)", "Variable (SG&A)",
            case_marketing, m_mkt,
            "Social media (TikTok/Instagram) & community marketing spend",
        ),
        _rec(
            "SGA", "SG&A", "Platform Dev Allocation (New)",
            "Platform build & migration allocation (Months 1–6)", "Leveled Dev (M1–M6)",
            0.0, m_plat_dev,
            "Pay-as-needed 6-month sprint drawdown + M4/M5/M6 holdback + expected contingency (~999.9M total)",
        ),
        _rec(
            "SGA", "SG&A", "Platform & Community OpEx (New)",
            "Platform cloud, OTP, maintenance & uni ops (Month 5+)", "Recurring OpEx (M5+)",
            0.0, m_plat_rec,
            "16.5M/mo in M5–M6 -> 26.5M/mo M7+ (Cloud, OTP, SaaS, Uni Ops & Maintenance Retainer)",
        ),
        _rec(
            "SGA", "SG&A", "External Legal/Noise & U-18 Safety (New)",
            "Ward noise-complaint legal handling & U-18 safety (externals.toml)", "External Risk (SG&A)",
            0.0, m_ext_leg_saf,
            "Post-21:00 ward noise mediation/acoustic upkeep (~6.5M–9.4M/mo) + U-18 safety compliance (~3.2M–4.3M/mo)",
        ),
        _rec(
            "SUMMARY", "SG&A", "SUBTOTAL SG&A",
            "Subtotal Selling, General, Admin, Platform & External Compliance", "Subtotal",
            case_total_sga, m_tot_sga,
            "Consolidated 2-club payroll, admin, marketing, platform build/OpEx & legal/safety compliance",
        ),

        _rec(
            "SUMMARY", "FS-Month", "OPEX + SG&A",
            "Total Operating Expenses (Facility OpEx + SG&A)", "Subtotal",
            case_total_opex + case_total_sga, m_tot_opex_sga,
            "Consolidated Facility OpEx + SG&A across 20 courts and digital platform",
        ),
        _rec(
            "SUMMARY", "FS-Month", "TOTAL ALL COSTS",
            "Total Monthly Operating Costs (COGS + OpEx + SG&A)", "Subtotal",
            case_total_costs, m_tot_costs,
            "All direct variable COGS, facility OpEx, SG&A, platform & external compliance costs",
        ),
        _rec(
            "SUMMARY", "FS-Month", "OPERATING PROFIT",
            f"Net Operating Income (EBITDA) [5Y Margin = {(m_ebitda.sum()/tot_5y_rev)*100:.2f}%]", "Operating Profit",
            case_ebitda, m_ebitda,
            f"Y3 Annual EBITDA = {mc.y3_annual_ebitda_mean_mvnd:,.1f}M VND (vs Investor Minh Tran 3,500M target)",
        ),
    )
    line_items_df = pd.DataFrame([vars(r) for r in line_items])

    # 2. 60-MONTH (5-YEAR) BALANCE SHEET & CASH FLOW STATEMENT RECONCILIATION
    #    Incorporates:
    #      - Months 7–10 (Jul-27 to Oct-27): Evenly split Fit-Out CapEx = -1,750.0M VND/month (-7,000.0M total)
    #      - Month 7 (Jul-27, Q3 2027 Expansion Kickoff): Investor Minh Tran Equity Injection = +8,000.0M VND
    #        (7,000.0M for Venue 2 fit-out + 1,000.0M added to PICK! cash reserve per Case Page 3 & 4)
    bs = cfg.balance_sheet
    vs_cfg = cfg.venue_scaling
    build_s_idx = vs_cfg.build_start_month_index - 1
    build_e_idx = vs_cfg.build_end_month_index - 1
    inv_sign_m_idx = vs_cfg.investor_signing_tranche_month_index - 1
    inv_m_idx = vs_cfg.investor_equity_injection_month_index - 1

    start_cash_arr = np.zeros(n_months)
    ocf_arr = np.zeros(n_months)
    icf_arr = np.zeros(n_months)
    fcf_arr = np.zeros(n_months)
    net_cf_arr = np.zeros(n_months)
    end_cash_arr = np.zeros(n_months)
    end_cash_case_arr = np.zeros(n_months)

    inv_arr = np.full(n_months, bs.inventory_mvnd)
    ar_arr = np.full(n_months, bs.accounts_receivable_mvnd)
    prepaid_arr = np.full(n_months, bs.prepaid_expenses_mvnd)
    tot_curr_assets_arr = np.zeros(n_months)

    ppe_net_arr = np.zeros(n_months)
    intang_arr = np.full(n_months, bs.intangible_assets_mvnd)
    tot_lt_assets_arr = np.zeros(n_months)
    tot_assets_arr = np.zeros(n_months)

    ap_arr = np.full(n_months, bs.accounts_payable_mvnd)
    accrued_arr = np.full(n_months, bs.accrued_expenses_payable_mvnd)
    st_debt_arr = np.full(n_months, bs.short_term_debt_mvnd)
    tot_curr_liab_arr = ap_arr + accrued_arr + st_debt_arr
    lt_debt_arr = np.full(n_months, bs.long_term_debt_mvnd)
    tot_liab_arr = tot_curr_liab_arr + lt_debt_arr

    contrib_cap_arr = np.zeros(n_months)
    ret_earn_arr = np.zeros(n_months)
    tot_equity_arr = np.zeros(n_months)
    tot_liab_and_eq_arr = np.zeros(n_months)
    bs_check_arr = np.zeros(n_months)

    curr_cash = bs.starting_cash_mvnd
    curr_cash_case = bs.starting_cash_mvnd
    curr_ppe = bs.equipment_ppe_gross_mvnd
    curr_contrib = bs.contributed_capital_mvnd
    curr_re = bs.starting_retained_earnings_mvnd

    cash_table_rows = []

    for m_idx in range(n_months):
        start_cash_arr[m_idx] = curr_cash
        ocf = float(m_net_inc[m_idx] + m_da[m_idx])

        capex_m = (
            vs_cfg.even_monthly_fitout_capex_mvnd
            if (build_s_idx <= m_idx <= build_e_idx)
            else bs.monthly_capex_mvnd
        )
        icf = -capex_m

        if m_idx == inv_sign_m_idx:
            eq_m = vs_cfg.investor_signing_tranche_mvnd
        elif m_idx == inv_m_idx:
            eq_m = vs_cfg.investor_expansion_tranche_mvnd
        else:
            eq_m = bs.monthly_equity_funding_mvnd
        fcf = eq_m - bs.monthly_dividends_mvnd

        net_cf = ocf + icf + fcf
        curr_cash = curr_cash + net_cf

        curr_cash_case = float(mc.base_ending_cash_mean[m_idx])

        ocf_arr[m_idx] = ocf
        icf_arr[m_idx] = icf
        fcf_arr[m_idx] = fcf
        net_cf_arr[m_idx] = net_cf
        end_cash_arr[m_idx] = curr_cash
        end_cash_case_arr[m_idx] = curr_cash_case

        curr_ppe = curr_ppe + capex_m - da
        curr_contrib = curr_contrib + eq_m
        curr_re = curr_re + float(m_net_inc[m_idx]) - bs.monthly_dividends_mvnd

        ppe_net_arr[m_idx] = curr_ppe
        tot_curr_assets_arr[m_idx] = curr_cash + inv_arr[m_idx] + ar_arr[m_idx] + prepaid_arr[m_idx]
        tot_lt_assets_arr[m_idx] = curr_ppe + intang_arr[m_idx]
        tot_assets_arr[m_idx] = tot_curr_assets_arr[m_idx] + tot_lt_assets_arr[m_idx]

        contrib_cap_arr[m_idx] = curr_contrib
        ret_earn_arr[m_idx] = curr_re
        tot_equity_arr[m_idx] = curr_contrib + curr_re
        tot_liab_and_eq_arr[m_idx] = tot_liab_arr[m_idx] + tot_equity_arr[m_idx]
        bs_check_arr[m_idx] = tot_assets_arr[m_idx] - tot_liab_and_eq_arr[m_idx]

        cash_table_rows.append(
            {
                "month_index": m_idx + 1,
                "month_label": month_names[m_idx],
                "year_num": (m_idx // 12) + 1,
                "active_courts": int(venue_scaling.active_courts_60m[m_idx]),
                "starting_cash_mvnd": round(float(start_cash_arr[m_idx]), 2),
                "operating_cash_flow_ocf_mvnd": round(float(ocf), 2),
                "venue_fitout_capex_icf_mvnd": round(float(icf), 2),
                "investor_equity_fcf_mvnd": round(float(fcf), 2),
                "net_cash_flow_mvnd": round(float(net_cf), 2),
                "ending_cash_balance_mean_mvnd": round(float(curr_cash), 2),
                "ending_cash_balance_p05_mvnd": round(float(mc.ending_cash_p05[m_idx]), 2),
                "ending_cash_balance_p95_mvnd": round(float(mc.ending_cash_p95[m_idx]), 2),
                "stress_30pct_ending_cash_mvnd": round(float(mc.stress_ending_cash_mean[m_idx]), 2),
                "case_baseline_ending_cash_mvnd": round(float(curr_cash_case), 2),
                "case_baseline_ending_cash_p05_mvnd": round(float(mc.base_ending_cash_p05[m_idx]), 2),
                "case_baseline_ending_cash_p95_mvnd": round(float(mc.base_ending_cash_p95[m_idx]), 2),
            }
        )

    cash_flow_60m_df = pd.DataFrame(cash_table_rows)

    # Row specifications for 3-Statement Table:
    three_stmt_rows: list[tuple[str, str, str, str, str, float | None, np.ndarray | None]] = [
        # ==================== STATEMENT 1: INCOME STATEMENT ====================
        ("IS_HDR", "INCOME STATEMENT", "STATEMENT 1 — INCOME STATEMENT (PROFIT & LOSS, M VND)", "header", "none", None, None),
        ("MEM_COURTS", "INCOME STATEMENT", "  Memo: Active Courts in Operation (6 -> 20 in M14)", "input_linked", "end_stock", 6.0, venue_scaling.active_courts_60m.astype(float)),
        ("MEM_CNT", "INCOME STATEMENT", "  Memo: Active Paying Members (PICK! Social @ 399k/mo)", "input_linked", "end_stock", 0.0, mc.active_subscribers_mean),
        ("REV_1A", "INCOME STATEMENT", "  Revenue: Court hire — Venue 1 (6 Courts, AdaptivePricing)", "input_linked", "flow", case_court_hire, m_v1_court),
        ("REV_1B", "INCOME STATEMENT", "  Revenue: Court hire — Venue 2 (14 Courts, Opens M14 / Feb-28)", "input_linked", "flow", 0.0, m_v2_court),
        ("REV_2", "INCOME STATEMENT", "  Revenue: Membership subscriptions (Month 5+)", "input_linked", "flow", 0.0, m_mem_rev),
        ("REV_3", "INCOME STATEMENT", "  Revenue: Paddle rental and balls (Scales to 20 Courts)", "input", "flow", paddle_balls, m_paddle),
        ("REV_4", "INCOME STATEMENT", "  Revenue: Coaching (Scales to 20 Courts + Studio)", "input", "flow", coaching, m_coach_rev),
        ("REV_5", "INCOME STATEMENT", "  Revenue: Food and drinks (Scales to 20 Courts + Cafe)", "input", "flow", food_drinks, m_fnb_rev),
        ("REV_TOT", "INCOME STATEMENT", "Total Revenue", "subtotal", "flow", case_total_rev, m_tot_rev),

        ("COGS_1", "INCOME STATEMENT", "  COGS: Coach fees (paid per lesson)", "input", "flow", case_coach_fees, m_coach_cogs),
        ("COGS_2", "INCOME STATEMENT", "  COGS: Food and drink supplies", "input", "flow", case_fb_supplies, m_fnb_cogs),
        ("COGS_3", "INCOME STATEMENT", "  COGS: Payment gateway fees (~1.50% Rev) & consumables", "input_linked", "flow", case_pay_consumables, m_pay_cons),
        ("COGS_4", "INCOME STATEMENT", "  COGS: Membership partner courts (@ 200k/hr) & events", "input_linked", "flow", 0.0, m_mem_event),
        ("COGS_TOT", "INCOME STATEMENT", "Total Cost of Goods Sold (COGS)", "subtotal", "flow", case_total_cogs, m_tot_cogs),

        ("GP_TOT", "INCOME STATEMENT", "Gross Profit", "total", "flow", case_gross_profit, m_gp),
        ("GP_PCT", "INCOME STATEMENT", "  Gross Margin (%)", "pct", "pct", (case_gross_profit / case_total_rev) * 100.0, m_gp_pct),

        ("OPEX_1A", "INCOME STATEMENT", "  OpEx: Venue 1 lease (6 Courts, 150M/mo)", "input", "flow", case_venue_lease, m_v1_lease),
        ("OPEX_1B", "INCOME STATEMENT", "  OpEx: Venue 2 lease (14 Courts, 330M/mo from M10 / Oct-27)", "input_linked", "flow", 0.0, m_v2_lease),
        ("OPEX_2", "INCOME STATEMENT", "  OpEx: Electricity, lighting and water (Venue 1 + Venue 2)", "input", "flow", case_utilities, m_util),
        ("OPEX_3", "INCOME STATEMENT", "  OpEx: Maintenance and insurance (Venue 1 + Venue 2)", "input", "flow", case_maint_ins, m_maint),
        ("OPEX_TOT", "INCOME STATEMENT", "Subtotal Facility Operating Expenses (OpEx)", "subtotal", "flow", case_total_opex, m_tot_opex),

        ("SGA_1", "INCOME STATEMENT", "  SG&A: Staff wages (Venue 1 130M + Venue 2 215M from M14)", "input", "flow", case_staff_wages, m_staff),
        ("SGA_2", "INCOME STATEMENT", "  SG&A: Booking system and administration (Base)", "input", "flow", case_booking_admin, m_admin),
        ("SGA_3", "INCOME STATEMENT", "  SG&A: Marketing (Growth & Social)", "input", "flow", case_marketing, m_mkt),
        ("SGA_4", "INCOME STATEMENT", "  SG&A: Platform build & migration allocation (M1–M6)", "input_linked", "flow", 0.0, m_plat_dev),
        ("SGA_5", "INCOME STATEMENT", "  SG&A: Platform cloud, OTP, maint & uni ops (M5+)", "input_linked", "flow", 0.0, m_plat_rec),
        ("SGA_6", "INCOME STATEMENT", "  SG&A: Legal/ward noise handling & U-18 safety (externals.toml)", "input_linked", "flow", 0.0, m_ext_leg_saf),
        ("SGA_TOT", "INCOME STATEMENT", "Subtotal Selling, General, Admin & Platform", "subtotal", "flow", case_total_sga, m_tot_sga),

        ("OPEX_SGA_TOT", "INCOME STATEMENT", "Total Operating Expenses (OpEx + SG&A)", "subtotal", "flow", case_total_opex + case_total_sga, m_tot_opex_sga),
        ("ALL_COST_TOT", "INCOME STATEMENT", "Total Costs (COGS + OpEx + SG&A)", "subtotal", "flow", case_total_costs, m_tot_costs),

        ("EBITDA_TOT", "INCOME STATEMENT", "Net Operating Income (EBITDA)", "grand_total", "flow", case_ebitda, m_ebitda),
        ("EBITDA_PCT", "INCOME STATEMENT", "  EBITDA Margin (%)", "pct", "pct", (case_ebitda / case_total_rev) * 100.0, m_ebitda_pct),
        ("DA_EXP", "INCOME STATEMENT", "  Depreciation & Amortization (D&A)", "formula", "flow", da, m_da),
        ("INT_EXP", "INCOME STATEMENT", "  Interest & Tax Expense", "formula", "flow", interest + case_taxes, m_int + m_taxes),
        ("NET_INC", "INCOME STATEMENT", "Net Income (Profit)", "grand_total", "flow", case_net_income, m_net_inc),

        # ==================== STATEMENT 2: BALANCE SHEET ====================
        ("BS_HDR", "BALANCE SHEET", "STATEMENT 2 — BALANCE SHEET (END OF PERIOD, M VND)", "header", "none", None, None),
        ("BS_CASH", "BALANCE SHEET", "  Cash and Cash Equivalents (Includes 500M M1 + 500M M10 Reserve)", "formula", "end_stock", bs.starting_cash_mvnd + case_net_income, end_cash_arr),
        ("BS_INV", "BALANCE SHEET", "  Inventory (F&B & Balls/Paddles)", "input", "end_stock", bs.inventory_mvnd, inv_arr),
        ("BS_PRE", "BALANCE SHEET", "  Prepaid Expenses & Lease Deposit", "input", "end_stock", bs.prepaid_expenses_mvnd, prepaid_arr),
        ("BS_CA_TOT", "BALANCE SHEET", "Total Current Assets", "subtotal", "end_stock", bs.starting_cash_mvnd + case_net_income + bs.inventory_mvnd + bs.prepaid_expenses_mvnd, tot_curr_assets_arr),
        ("BS_PPE", "BALANCE SHEET", "  Equipment & Venue Fit-Out PP&E (+7,000M in M10–M13)", "input_linked", "end_stock", bs.equipment_ppe_gross_mvnd, ppe_net_arr),
        ("BS_INTANG", "BALANCE SHEET", "  Intangible Assets (Platform CapEx)", "input", "end_stock", bs.intangible_assets_mvnd, intang_arr),
        ("BS_LTA_TOT", "BALANCE SHEET", "Total Long-Term (Fixed) Assets", "subtotal", "end_stock", bs.equipment_ppe_gross_mvnd + bs.intangible_assets_mvnd, tot_lt_assets_arr),
        ("BS_ASSETS_TOT", "BALANCE SHEET", "TOTAL ASSETS", "total", "end_stock", bs.starting_cash_mvnd + case_net_income + bs.inventory_mvnd + bs.prepaid_expenses_mvnd + bs.equipment_ppe_gross_mvnd, tot_assets_arr),

        ("BS_AP", "BALANCE SHEET", "  Accounts Payable (Supplies & Gateway)", "input", "end_stock", bs.accounts_payable_mvnd, ap_arr),
        ("BS_ACC", "BALANCE SHEET", "  Accrued Coaching & Staff Payables", "input", "end_stock", bs.accrued_expenses_payable_mvnd, accrued_arr),
        ("BS_LIAB_TOT", "BALANCE SHEET", "TOTAL LIABILITIES", "subtotal", "end_stock", bs.accounts_payable_mvnd + bs.accrued_expenses_payable_mvnd, tot_liab_arr),

        ("BS_CAP", "BALANCE SHEET", "  Contributed Capital (+500M M1 & +7,500M M10 Investor Equity)", "input_linked", "end_stock", bs.contributed_capital_mvnd, contrib_cap_arr),
        ("BS_RE", "BALANCE SHEET", "  Retained Earnings (Cumulative)", "formula", "end_stock", bs.starting_retained_earnings_mvnd + case_net_income, ret_earn_arr),
        ("BS_EQ_TOT", "BALANCE SHEET", "TOTAL OWNER'S EQUITY", "subtotal", "end_stock", bs.contributed_capital_mvnd + bs.starting_retained_earnings_mvnd + case_net_income, tot_equity_arr),
        ("BS_LE_TOT", "BALANCE SHEET", "TOTAL LIABILITIES & OWNER'S EQUITY", "total", "end_stock", tot_liab_arr[0] + bs.contributed_capital_mvnd + bs.starting_retained_earnings_mvnd + case_net_income, tot_liab_and_eq_arr),
        ("BS_CHK", "BALANCE SHEET", "  Balance Sheet Check (Assets - L&E)", "formula", "end_stock", 0.0, bs_check_arr),

        # ==================== STATEMENT 3: CASH FLOW STATEMENT ====================
        ("CF_HDR", "CASH FLOW", "STATEMENT 3 — CASH FLOW STATEMENT (M VND)", "header", "none", None, None),
        ("CF_START", "CASH FLOW", "Starting Cash Balance", "formula", "start_stock", bs.starting_cash_mvnd, start_cash_arr),
        ("CF_NI", "CASH FLOW", "  (+) Net Income (Profit)", "formula", "flow", case_net_income, m_net_inc),
        ("CF_DA", "CASH FLOW", "  (+) Depreciation & Amortization (D&A)", "formula", "flow", da, m_da),
        ("CF_NWC", "CASH FLOW", "  (-) Change in Working Capital", "formula", "flow", 0.0, np.zeros(n_months)),
        ("CF_OCF", "CASH FLOW", "Total Operating Cash Flow (OCF)", "subtotal", "flow", case_net_income, ocf_arr),
        ("CF_CAPEX", "CASH FLOW", "  (-) Venue 2 Fit-Out CapEx (-1,750M/mo in M10–M13 = -7,000M)", "input_linked", "flow", -bs.monthly_capex_mvnd, icf_arr),
        ("CF_ICF", "CASH FLOW", "Total Investing Cash Flow (ICF)", "subtotal", "flow", -bs.monthly_capex_mvnd, icf_arr),
        ("CF_EQ", "CASH FLOW", "  (+) Investor Minh Tran Equity (+500M M1 & +7,500M M10 = +8,000M)", "input_linked", "flow", bs.monthly_equity_funding_mvnd - bs.monthly_dividends_mvnd, fcf_arr),
        ("CF_FCF", "CASH FLOW", "Total Financing Cash Flow (FCF)", "subtotal", "flow", bs.monthly_equity_funding_mvnd - bs.monthly_dividends_mvnd, fcf_arr),
        ("CF_NET", "CASH FLOW", "Net Cash Flow (Change in Cash)", "total", "flow", case_net_income, net_cf_arr),
        ("CF_END", "CASH FLOW", "ENDING CASH BALANCE (Model)", "grand_total", "end_stock", bs.starting_cash_mvnd + case_net_income, end_cash_arr),
        ("CF_END_CASE", "CASH FLOW", "  Memo: Case Baseline Ending Cash Balance", "formula", "end_stock", bs.starting_cash_mvnd + case_net_income, end_cash_case_arr),
    ]

    # Build FS-Month DataFrame (60 Months + FY1..FY5 + 5Y_Total + Pct_of_Rev)
    month_records = []
    for code, stmt, label, rtype, agg_mode, case_m1, arr in three_stmt_rows:
        unit_str = (
            "Courts"
            if code == "MEM_COURTS"
            else (
                "Members"
                if code == "MEM_CNT"
                else ("%" if rtype == "pct" else ("M VND" if rtype != "header" else ""))
            )
        )
        row_dict: dict[str, object] = {
            "row_code": code,
            "statement": stmt,
            "line_item": label,
            "row_type": rtype,
            "agg_mode": agg_mode,
            "unit": unit_str,
            "Case_M1_Baseline": round(float(case_m1), 4) if case_m1 is not None else None,
        }
        if arr is None:
            for m in month_names:
                row_dict[m] = None
            for y_idx in range(1, 6):
                row_dict[f"FY{y_idx}_Total"] = None
            row_dict["5Y_Total"] = None
            row_dict["Pct_of_Rev"] = None
        else:
            for m_idx, m in enumerate(month_names):
                row_dict[m] = round(float(arr[m_idx]), 4)

            for y_idx in range(1, 6):
                s_i = (y_idx - 1) * 12
                e_i = y_idx * 12
                if agg_mode == "flow":
                    row_dict[f"FY{y_idx}_Total"] = round(float(arr[s_i:e_i].sum()), 4)
                elif agg_mode == "pct":
                    if code == "GP_PCT":
                        row_dict[f"FY{y_idx}_Total"] = round(float(m_gp[s_i:e_i].sum() / m_tot_rev[s_i:e_i].sum() * 100.0), 4)
                    elif code == "EBITDA_PCT":
                        row_dict[f"FY{y_idx}_Total"] = round(float(m_ebitda[s_i:e_i].sum() / m_tot_rev[s_i:e_i].sum() * 100.0), 4)
                    else:
                        row_dict[f"FY{y_idx}_Total"] = round(float(arr[s_i:e_i].mean()), 4)
                elif agg_mode == "start_stock":
                    row_dict[f"FY{y_idx}_Total"] = round(float(arr[s_i]), 4)
                elif agg_mode == "end_stock":
                    row_dict[f"FY{y_idx}_Total"] = round(float(arr[e_i - 1]), 4)

            if agg_mode == "flow":
                tot5 = float(arr.sum())
                row_dict["5Y_Total"] = round(tot5, 4)
                row_dict["Pct_of_Rev"] = round((tot5 / tot_5y_rev) * 100.0, 4) if stmt == "INCOME STATEMENT" else None
            elif agg_mode == "pct":
                if code == "GP_PCT":
                    val5 = float(m_gp.sum() / tot_5y_rev * 100.0)
                elif code == "EBITDA_PCT":
                    val5 = float(m_ebitda.sum() / tot_5y_rev * 100.0)
                else:
                    val5 = float(arr.mean())
                row_dict["5Y_Total"] = round(val5, 4)
                row_dict["Pct_of_Rev"] = round(val5, 4)
            elif agg_mode == "start_stock":
                row_dict["5Y_Total"] = round(float(arr[0]), 4)
                row_dict["Pct_of_Rev"] = None
            elif agg_mode == "end_stock":
                row_dict["5Y_Total"] = round(float(arr[-1]), 4)
                row_dict["Pct_of_Rev"] = None

        month_records.append(row_dict)

    fs_month_df = pd.DataFrame(month_records)

    # Build FS-Annual DataFrame (FY 2027 to FY 2031 + 5Y_Total)
    year_labels = (
        "FY 2027 (Y1)",
        "FY 2028 (Y2)",
        "FY 2029 (Y3)",
        "FY 2030 (Y4)",
        "FY 2031 (Y5)",
    )
    annual_records = []
    for m_rec in month_records:
        row_dict = {
            "row_code": m_rec["row_code"],
            "statement": m_rec["statement"],
            "line_item": m_rec["line_item"],
            "row_type": m_rec["row_type"],
            "agg_mode": m_rec["agg_mode"],
            "unit": m_rec["unit"],
            "Case_M1_Baseline": m_rec["Case_M1_Baseline"],
            "Model_M1_Build": m_rec["Jan-27"],
            "Model_M5_Launch": m_rec["May-27"],
            "Model_M11_Open20": m_rec["Nov-27"],
        }
        for y_idx, y_lbl in enumerate(year_labels, 1):
            row_dict[y_lbl] = m_rec[f"FY{y_idx}_Total"]
        row_dict["5Y_Total"] = m_rec["5Y_Total"]
        row_dict["Pct_of_Rev"] = m_rec["Pct_of_Rev"]
        annual_records.append(row_dict)

    fs_annual_df = pd.DataFrame(annual_records)

    ann_rev = np.array([m_tot_rev[(y * 12):((y + 1) * 12)].sum() for y in range(5)])
    ann_costs = np.array([m_tot_costs[(y * 12):((y + 1) * 12)].sum() for y in range(5)])
    ann_ebitda = np.array([m_ebitda[(y * 12):((y + 1) * 12)].sum() for y in range(5)])
    ann_ni = np.array([m_net_inc[(y * 12):((y + 1) * 12)].sum() for y in range(5)])

    min_cash_idx = int(np.argmin(end_cash_arr))

    return FinancialModelResults(
        adaptive=adaptive,
        timeline_cost=timeline_cost,
        venue_scaling=venue_scaling,
        marketing=marketing,
        mc=mc,
        regimes=regimes,
        total_available_court_hours=total_avail_hrs,
        total_booked_court_hours=total_booked_hrs,
        blended_utilisation_rate=blended_util,
        blended_court_rate_kvnd=blended_rate_kvnd,
        unique_customers=cfg.customers.unique_customers_per_month,
        exact_player_visits=exact_visits,
        reported_player_visits=cfg.customers.approx_player_visits_per_month,
        weighted_avg_spend_per_visit_kvnd=weighted_spend_kvnd,
        case_court_hire_mvnd=case_court_hire,
        case_total_revenue_mvnd=case_total_rev,
        case_total_cogs_mvnd=case_total_cogs,
        case_gross_profit_mvnd=case_gross_profit,
        case_total_opex_mvnd=case_total_opex,
        case_total_sga_mvnd=case_total_sga,
        case_total_costs_mvnd=case_total_costs,
        case_ebitda_mvnd=case_ebitda,
        case_net_income_mvnd=case_net_income,
        court_hire_mvnd=float(m_court.mean()),
        venue1_court_hire_m1_mvnd=float(m_v1_court[0]),
        venue2_court_hire_steady_mvnd=float(court_hire_base * cfg.venue_scaling.capacity_ratio_new_vs_existing),
        paddle_balls_mvnd=float(m_paddle.mean()),
        coaching_mvnd=float(m_coach_rev.mean()),
        food_drinks_mvnd=float(m_fnb_rev.mean()),
        m24_membership_rev_mvnd=float(m_mem_rev[23]),
        five_year_membership_rev_mvnd=float(m_mem_rev.sum()),
        monthly_active_courts_arr=venue_scaling.active_courts_60m,
        monthly_trend_index_arr=mc.trend_index_mean,
        monthly_venue1_court_hire_arr=m_v1_court,
        monthly_venue2_court_hire_arr=m_v2_court,
        monthly_court_hire_arr=m_court,
        monthly_membership_rev_arr=m_mem_rev,
        monthly_paddle_balls_arr=m_paddle,
        monthly_coaching_arr=m_coach_rev,
        monthly_food_drinks_arr=m_fnb_rev,
        monthly_total_rev_arr=m_tot_rev,
        monthly_coach_fees_arr=m_coach_cogs,
        monthly_fnb_supplies_arr=m_fnb_cogs,
        monthly_payment_consumables_arr=m_pay_cons,
        monthly_membership_event_cogs_arr=m_mem_event,
        monthly_total_cogs_arr=m_tot_cogs,
        monthly_gross_profit_arr=m_gp,
        monthly_gross_margin_pct_arr=m_gp_pct,
        monthly_venue1_lease_arr=m_v1_lease,
        monthly_venue2_lease_arr=m_v2_lease,
        monthly_venue_lease_arr=m_lease,
        monthly_utilities_arr=m_util,
        monthly_maint_ins_arr=m_maint,
        monthly_total_opex_arr=m_tot_opex,
        monthly_venue1_staff_wages_arr=m_v1_staff,
        monthly_venue2_staff_wages_arr=m_v2_staff,
        monthly_staff_wages_arr=m_staff,
        monthly_booking_admin_arr=m_admin,
        monthly_marketing_arr=m_mkt,
        monthly_platform_dev_alloc_arr=m_plat_dev,
        monthly_platform_recurring_opex_arr=m_plat_rec,
        monthly_external_legal_safety_sga_arr=m_ext_leg_saf,
        monthly_total_sga_arr=m_tot_sga,
        monthly_total_opex_and_sga_arr=m_tot_opex_sga,
        monthly_total_costs_arr=m_tot_costs,
        monthly_ebitda_arr=m_ebitda,
        monthly_ebitda_margin_pct_arr=m_ebitda_pct,
        monthly_net_income_arr=m_net_inc,
        five_year_revenue_mvnd=tot_5y_rev,
        five_year_cogs_mvnd=float(m_tot_cogs.sum()),
        five_year_gross_profit_mvnd=float(m_gp.sum()),
        five_year_opex_mvnd=float(m_tot_opex.sum()),
        five_year_sga_mvnd=float(m_tot_sga.sum()),
        five_year_total_costs_mvnd=float(m_tot_costs.sum()),
        five_year_ebitda_mvnd=float(m_ebitda.sum()),
        five_year_net_income_mvnd=float(m_net_inc.sum()),
        annual_revenue_arr=ann_rev,
        annual_total_costs_arr=ann_costs,
        annual_ebitda_arr=ann_ebitda,
        annual_net_income_arr=ann_ni,
        starting_cash_mvnd=bs.starting_cash_mvnd,
        min_cash_balance_model_mvnd=float(end_cash_arr[min_cash_idx]),
        min_cash_month_label=month_names[min_cash_idx],
        m6_pre_expansion_cash_mvnd=float(end_cash_arr[5]),
        m10_post_build_cash_mvnd=float(end_cash_arr[9]),
        ending_cash_60m_model_mvnd=float(end_cash_arr[-1]),
        ending_cash_60m_case_mvnd=float(end_cash_case_arr[-1]),
        monthly_starting_cash_model=start_cash_arr,
        monthly_ocf_model=ocf_arr,
        monthly_icf_capex_model=icf_arr,
        monthly_fcf_equity_model=fcf_arr,
        monthly_net_cash_flow_model=net_cf_arr,
        monthly_ending_cash_model=end_cash_arr,
        monthly_ending_cash_case=end_cash_case_arr,
        line_items=line_items,
        line_items_df=line_items_df,
        fs_month_df=fs_month_df,
        fs_annual_df=fs_annual_df,
        cash_flow_60m_df=cash_flow_60m_df,
        month_labels=tuple(month_names),
        year_labels=year_labels,
    )
