"""Configuration loader and typed distribution schema for PICK! 3-Statement Financial Model (central `configs/`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
import numpy as np


@dataclass(frozen=True)
class DistParam:
    """Truncated Normal distribution parameter specification."""

    dist: str
    mean: float
    std: float
    min: float
    max: float
    unit: str = ""
    source: str = ""

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """Sample from the truncated normal distribution using numpy Generator."""
        if self.std <= 0.0 or self.min >= self.max:
            return np.full(size, self.mean, dtype=float)
        draws = rng.normal(loc=self.mean, scale=self.std, size=size)
        return np.clip(draws, self.min, self.max)


def _parse_dist(d: dict) -> DistParam:
    return DistParam(
        dist=str(d.get("dist", "truncated_normal")),
        mean=float(d["mean"]),
        std=float(d.get("std", 0.0)),
        min=float(d.get("min", d["mean"])),
        max=float(d.get("max", d["mean"])),
        unit=str(d.get("unit", "")),
        source=str(d.get("source", "")),
    )


@dataclass(frozen=True)
class CompanyConfig:
    name: str
    currency: str
    display_unit: str
    vnd_per_million: int
    model_start_month: str
    projection_months: int
    projection_years: int
    has_actuals: bool
    use_adaptive_pricing_output: bool
    adaptive_pricing_output_dir: str
    timeline_output_dir: str
    venue_scaling_output_dir: str
    marketing_output_dir: str


@dataclass(frozen=True)
class OperationsConfig:
    courts_count: int
    open_hour: int
    close_hour: int
    days_per_month: int
    weekdays_per_month: int
    weekend_days_per_month: int
    weekday_daytime_hours_per_day: int
    weekday_daytime_price_kvnd: float
    weekday_daytime_utilisation: float
    weekday_evening_hours_per_day: int
    weekday_evening_price_kvnd: float
    weekday_evening_utilisation: float
    weekend_hours_per_day: int
    weekend_price_kvnd: float
    weekend_utilisation: float

    @property
    def operating_hours_per_day(self) -> int:
        return self.close_hour - self.open_hour


@dataclass(frozen=True)
class CustomerSegmentConfig:
    key: str
    label: str
    main_need: str
    share_dist: DistParam
    visits_per_month_dist: DistParam
    spend_per_visit_kvnd_dist: DistParam
    subscription_propensity_dist: DistParam

    @property
    def share(self) -> float:
        return self.share_dist.mean

    @property
    def visits_per_month(self) -> float:
        return self.visits_per_month_dist.mean

    @property
    def spend_per_visit_kvnd(self) -> float:
        return self.spend_per_visit_kvnd_dist.mean


@dataclass(frozen=True)
class SocialMediaConfig:
    initial_followers_dist: DistParam
    monthly_new_follow_rate_dist: DistParam
    monthly_unfollow_rate_dist: DistParam


@dataclass(frozen=True)
class SurveyExhibit3Config:
    expect_regular_play_3y_rate_dist: DistParam
    unsure_play_3y_rate_dist: DistParam
    no_play_3y_rate_dist: DistParam
    membership_stated_intent_rate_dist: DistParam
    intent_to_actual_follow_through_dist: DistParam


@dataclass(frozen=True)
class MembershipProgramConfig:
    launch_month_index: int        # 1-based: 5 = May-27 (after 4-month core build completes)
    launch_month_label: str        # "May-27"
    monthly_fee_kvnd: float        # 399.0 kVND
    monthly_fee_mvnd: float        # 0.399 M VND
    case_year2_target_members: float
    case_year2_target_rev_mvnd: float
    launch_conversion_of_stated_intent_dist: DistParam
    monthly_churn_rate_dist: DistParam
    monthly_new_subscriber_rate_dist: DistParam
    addressable_member_ceiling_dist: DistParam


@dataclass(frozen=True)
class CustomersConfig:
    unique_customers_dist: DistParam
    registered_accounts_dist: DistParam
    player_visits_anchor_dist: DistParam
    segments: tuple[CustomerSegmentConfig, ...]
    social_media: SocialMediaConfig
    survey_exhibit3: SurveyExhibit3Config
    membership: MembershipProgramConfig

    @property
    def unique_customers_per_month(self) -> int:
        return int(round(self.unique_customers_dist.mean))

    @property
    def approx_player_visits_per_month(self) -> int:
        return int(round(self.player_visits_anchor_dist.mean))

    @property
    def registered_accounts(self) -> int:
        return int(round(self.registered_accounts_dist.mean))

    @property
    def social_followers(self) -> int:
        return int(round(self.social_media.initial_followers_dist.mean))


@dataclass(frozen=True)
class RevenueConfig:
    case_baseline_court_hire_mvnd: float
    paddle_rental_and_balls_mvnd: float
    coaching_mvnd: float
    food_and_drinks_mvnd: float


@dataclass(frozen=True)
class COGSConfig:
    coach_fees_share_dist: DistParam
    coach_fees_monthly_dist: DistParam
    fnb_supplies_share_dist: DistParam
    fnb_supplies_monthly_dist: DistParam
    payment_fee_pct_of_revenue_dist: DistParam
    consumables_replacement_monthly_dist: DistParam
    baseline_payment_and_consumables_dist: DistParam
    partner_court_hourly_rate_kvnd_dist: DistParam
    membership_event_partner_court_per_member_kvnd_dist: DistParam

    @property
    def coach_fees_mvnd(self) -> float:
        return self.coach_fees_monthly_dist.mean

    @property
    def food_and_drink_supplies_mvnd(self) -> float:
        return self.fnb_supplies_monthly_dist.mean

    @property
    def payment_fees_and_consumables_mvnd(self) -> float:
        return self.baseline_payment_and_consumables_dist.mean


@dataclass(frozen=True)
class OpExConfig:
    venue_lease_dist: DistParam
    electricity_lighting_water_dist: DistParam
    maintenance_and_insurance_dist: DistParam

    @property
    def venue_lease_mvnd(self) -> float:
        return self.venue_lease_dist.mean

    @property
    def electricity_lighting_water_mvnd(self) -> float:
        return self.electricity_lighting_water_dist.mean

    @property
    def maintenance_and_insurance_mvnd(self) -> float:
        return self.maintenance_and_insurance_dist.mean


@dataclass(frozen=True)
class SGAConfig:
    staff_wages_dist: DistParam
    booking_system_and_admin_dist: DistParam
    marketing_dist: DistParam

    @property
    def staff_wages_mvnd(self) -> float:
        return self.staff_wages_dist.mean

    @property
    def booking_system_and_admin_mvnd(self) -> float:
        return self.booking_system_and_admin_dist.mean

    @property
    def marketing_mvnd(self) -> float:
        return self.marketing_dist.mean


@dataclass(frozen=True)
class Venue2OpExSGAConfig:
    lease_start_month_index: int
    operations_open_month_index: int
    venue_lease_dist: DistParam
    electricity_lighting_water_dist: DistParam
    maintenance_and_insurance_dist: DistParam
    staff_wages_dist: DistParam


@dataclass(frozen=True)
class PlatformMonthlyOpExConfig:
    activation_month_index: int
    post_hypercare_month_index: int
    cloud_hosting_db_redis_dist: DistParam
    zalo_zns_otp_notifications_dist: DistParam
    observability_cicd_security_saas_dist: DistParam
    ongoing_maintenance_patching_retainer_dist: DistParam
    school_university_community_ops_dist: DistParam
    task_buffer_utilization_dist: DistParam

    @property
    def launch_months_5_6_mean_mvnd(self) -> float:
        return (
            self.cloud_hosting_db_redis_dist.mean
            + self.zalo_zns_otp_notifications_dist.mean
            + self.observability_cicd_security_saas_dist.mean
            + self.school_university_community_ops_dist.mean
        )

    @property
    def steady_state_months_7_plus_mean_mvnd(self) -> float:
        return self.launch_months_5_6_mean_mvnd + self.ongoing_maintenance_patching_retainer_dist.mean


@dataclass(frozen=True)
class VenueScalingCapExConfig:
    enabled: bool
    existing_courts: int
    new_venue_courts: int
    total_expanded_courts: int
    capacity_ratio_new_vs_existing: float
    build_start_month_index: int
    build_duration_months: int
    build_end_month_index: int
    open_month_index: int
    total_fitout_capex_mvnd: float
    even_monthly_fitout_capex_mvnd: float
    investor_total_new_money_mvnd: float
    investor_reserve_mvnd: float
    investor_signing_tranche_month_index: int
    investor_signing_tranche_mvnd: float
    investor_equity_injection_month_index: int
    investor_expansion_tranche_mvnd: float
    new_venue_lease_monthly_mvnd: float
    lease_start_month_index: int
    fill_up_duration_months_dist: DistParam
    initial_open_utilization_fraction_dist: DistParam
    network_customer_catchment_expansion_dist: DistParam


@dataclass(frozen=True)
class ExternalsConfig:
    initial_trend_index: float
    min_cumulative_trend_floor: float
    max_cumulative_trend_ceiling: float
    community_retention_insulation: float
    monthly_interest_change_rate_dist: DistParam
    legal_and_noise_monthly_cost_dist: DistParam
    legal_venue2_scaling_multiplier_dist: DistParam
    competitor_non_member_drag_rate_dist: DistParam
    weather_utilization_multiplier_dist: DistParam
    input_cost_drift_rate_dist: DistParam
    under18_safety_monthly_cost_dist: DistParam


@dataclass(frozen=True)
class TaxesAndOtherConfig:
    corporate_tax_rate: float
    tax_consolidation_month: int
    vat_output_rate: float
    vat_input_rate: float
    venue2_depreciation_useful_life_months: int
    depreciation_amortization_mvnd: float
    interest_expense_mvnd: float


@dataclass(frozen=True)
class BalanceSheetConfig:
    starting_cash_mvnd: float
    inventory_mvnd: float
    accounts_receivable_mvnd: float
    prepaid_expenses_mvnd: float
    equipment_ppe_gross_mvnd: float
    intangible_assets_mvnd: float
    accounts_payable_mvnd: float
    accrued_expenses_payable_mvnd: float
    short_term_debt_mvnd: float
    long_term_debt_mvnd: float
    contributed_capital_mvnd: float
    starting_retained_earnings_mvnd: float
    monthly_capex_mvnd: float
    monthly_equity_funding_mvnd: float
    monthly_dividends_mvnd: float


@dataclass(frozen=True)
class MonteCarloRandomnessConfig:
    random_seed: int
    n_simulations: int
    train_fraction: float
    ci_low_percentile: float
    ci_high_percentile: float
    monthly_demand_macro_shock_dist: DistParam
    subscriber_acquisition_shock_dist: DistParam
    subscriber_churn_shock_dist: DistParam
    ancillary_revenue_shock_dist: DistParam
    payment_gateway_cash_mix_shock_dist: DistParam
    stress_demand_drop_fraction: float
    stress_demand_multiplier: float


@dataclass(frozen=True)
class FinancialModelConfig:
    config_dir: Path
    company: CompanyConfig
    operations: OperationsConfig
    customers: CustomersConfig
    revenue: RevenueConfig
    cogs: COGSConfig
    opex: OpExConfig
    sga: SGAConfig
    venue2_opex_sga: Venue2OpExSGAConfig
    platform_opex: PlatformMonthlyOpExConfig
    venue_scaling: VenueScalingCapExConfig
    externals: ExternalsConfig
    taxes_and_other: TaxesAndOtherConfig
    balance_sheet: BalanceSheetConfig
    randomness: MonteCarloRandomnessConfig

    @property
    def resolved_adaptive_pricing_output_dir(self) -> Path:
        return (self.config_dir / self.company.adaptive_pricing_output_dir).resolve()

    @property
    def resolved_timeline_output_dir(self) -> Path:
        return (self.config_dir / self.company.timeline_output_dir).resolve()

    @property
    def resolved_venue_scaling_output_dir(self) -> Path:
        return (self.config_dir / self.company.venue_scaling_output_dir).resolve()

    @property
    def resolved_marketing_output_dir(self) -> Path:
        return (self.config_dir / self.company.marketing_output_dir).resolve()


def load_config(config_path: Path | None = None) -> FinancialModelConfig:
    """
    Load and validate the central financial model configuration from `CaseComp/configs/`
    (`pricing.toml`, `customers.toml`, `cogs.toml`, `opex.toml`, `capex.toml`, `externals.toml`, `randomness.toml`).
    """
    if config_path is None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
    else:
        resolved = config_path.resolve()
        config_dir = resolved if resolved.is_dir() else resolved.parent
        if not (config_dir / "customers.toml").exists():
            config_dir = Path(__file__).resolve().parents[2] / "configs"

    with open(config_dir / "pricing.toml", "rb") as f:
        pricing_raw = tomllib.load(f)
    with open(config_dir / "customers.toml", "rb") as f:
        cust_raw = tomllib.load(f)
    with open(config_dir / "cogs.toml", "rb") as f:
        cogs_raw = tomllib.load(f)
    with open(config_dir / "opex.toml", "rb") as f:
        opex_raw = tomllib.load(f)
    with open(config_dir / "capex.toml", "rb") as f:
        capex_raw = tomllib.load(f)
    with open(config_dir / "externals.toml", "rb") as f:
        ext_raw = tomllib.load(f)
    with open(config_dir / "randomness.toml", "rb") as f:
        rand_raw = tomllib.load(f)

    comp_raw = pricing_raw["company"]
    comp = CompanyConfig(
        name=str(comp_raw["name"]),
        currency=str(comp_raw["currency"]),
        display_unit=str(comp_raw["display_unit"]),
        vnd_per_million=int(comp_raw["vnd_per_million"]),
        model_start_month=str(comp_raw["model_start_month"]),
        projection_months=int(comp_raw["projection_months"]),
        projection_years=int(comp_raw["projection_years"]),
        has_actuals=bool(comp_raw["has_actuals"]),
        use_adaptive_pricing_output=bool(comp_raw["use_adaptive_pricing_output"]),
        adaptive_pricing_output_dir=str(comp_raw["adaptive_pricing_output_dir"]),
        timeline_output_dir=str(comp_raw.get("timeline_output_dir", "../PickSocialProjectTimeline/outputs")),
        venue_scaling_output_dir=str(comp_raw.get("venue_scaling_output_dir", "../VenueScalingTimeline/outputs")),
        marketing_output_dir=str(comp_raw.get("marketing_output_dir", "../Marketing/outputs")),
    )

    c_ops = pricing_raw["court_operations"]
    b_pricing = pricing_raw["baseline_pricing"]
    u_dists = pricing_raw["utilization_distributions"]
    ops = OperationsConfig(
        courts_count=int(c_ops["courts_count"]),
        open_hour=int(c_ops["open_hour"]),
        close_hour=int(c_ops["close_hour"]),
        days_per_month=int(c_ops["days_per_month"]),
        weekdays_per_month=int(c_ops["weekdays_per_month"]),
        weekend_days_per_month=int(c_ops["weekend_days_per_month"]),
        weekday_daytime_hours_per_day=int(c_ops["weekday_daytime_hours_per_day"]),
        weekday_daytime_price_kvnd=float(b_pricing["off_peak_price_kvnd"]),
        weekday_daytime_utilisation=float(u_dists["weekday_daytime"]["mean"]),
        weekday_evening_hours_per_day=int(c_ops["weekday_evening_hours_per_day"]),
        weekday_evening_price_kvnd=float(b_pricing["peak_price_kvnd"]),
        weekday_evening_utilisation=float(u_dists["weekday_evening"]["mean"]),
        weekend_hours_per_day=int(c_ops["weekend_hours_per_day"]),
        weekend_price_kvnd=float(b_pricing["peak_price_kvnd"]),
        weekend_utilisation=float(u_dists["weekend"]["mean"]),
    )

    segs = []
    for k, v in cust_raw["segments"].items():
        segs.append(
            CustomerSegmentConfig(
                key=k,
                label=str(v["label"]),
                main_need=str(v.get("main_need", "")),
                share_dist=_parse_dist(v["share"]),
                visits_per_month_dist=_parse_dist(v["visits_per_month"]),
                spend_per_visit_kvnd_dist=_parse_dist(v["spend_per_visit_kvnd"]),
                subscription_propensity_dist=_parse_dist(v["subscription_propensity_weight"]),
            )
        )

    sm_raw = cust_raw["social_media"]
    social_media = SocialMediaConfig(
        initial_followers_dist=_parse_dist(sm_raw["initial_followers"]),
        monthly_new_follow_rate_dist=_parse_dist(sm_raw["monthly_new_follow_rate"]),
        monthly_unfollow_rate_dist=_parse_dist(sm_raw["monthly_unfollow_rate"]),
    )

    sv_raw = cust_raw["survey_exhibit3"]
    survey_ex3 = SurveyExhibit3Config(
        expect_regular_play_3y_rate_dist=_parse_dist(sv_raw["expect_regular_play_3y_rate"]),
        unsure_play_3y_rate_dist=_parse_dist(sv_raw["unsure_play_3y_rate"]),
        no_play_3y_rate_dist=_parse_dist(sv_raw["no_play_3y_rate"]),
        membership_stated_intent_rate_dist=_parse_dist(sv_raw["membership_stated_intent_rate"]),
        intent_to_actual_follow_through_dist=_parse_dist(sv_raw["intent_to_actual_follow_through_multiplier"]),
    )

    mp_raw = cust_raw["membership_program"]
    membership = MembershipProgramConfig(
        launch_month_index=int(mp_raw["launch_month_index"]),
        launch_month_label=str(mp_raw["launch_month_label"]),
        monthly_fee_kvnd=float(mp_raw["monthly_fee_kvnd"]),
        monthly_fee_mvnd=float(mp_raw["monthly_fee_mvnd"]),
        case_year2_target_members=float(mp_raw["case_year2_target_members"]),
        case_year2_target_rev_mvnd=float(mp_raw["case_year2_target_rev_mvnd"]),
        launch_conversion_of_stated_intent_dist=_parse_dist(mp_raw["launch_conversion_of_stated_intent"]),
        monthly_churn_rate_dist=_parse_dist(mp_raw["monthly_churn_rate"]),
        monthly_new_subscriber_rate_dist=_parse_dist(mp_raw["monthly_new_subscriber_rate"]),
        addressable_member_ceiling_dist=_parse_dist(mp_raw["addressable_member_ceiling"]),
    )

    cust = CustomersConfig(
        unique_customers_dist=_parse_dist(cust_raw["unique_customers_monthly"]),
        registered_accounts_dist=_parse_dist(cust_raw["registered_accounts_initial"]),
        player_visits_anchor_dist=_parse_dist(cust_raw["player_visits_per_month_anchor"]),
        segments=tuple(segs),
        social_media=social_media,
        survey_exhibit3=survey_ex3,
        membership=membership,
    )

    rev = RevenueConfig(**pricing_raw["baseline_revenue_lines"])

    cogs = COGSConfig(
        coach_fees_share_dist=_parse_dist(cogs_raw["coach_fees_share_of_coaching_rev"]),
        coach_fees_monthly_dist=_parse_dist(cogs_raw["coach_fees_monthly_mvnd"]),
        fnb_supplies_share_dist=_parse_dist(cogs_raw["fnb_supplies_share_of_fnb_rev"]),
        fnb_supplies_monthly_dist=_parse_dist(cogs_raw["fnb_supplies_monthly_mvnd"]),
        payment_fee_pct_of_revenue_dist=_parse_dist(cogs_raw["payment_fee_pct_of_revenue"]),
        consumables_replacement_monthly_dist=_parse_dist(cogs_raw["consumables_replacement_monthly_mvnd"]),
        baseline_payment_and_consumables_dist=_parse_dist(cogs_raw["baseline_payment_and_consumables_monthly_mvnd"]),
        partner_court_hourly_rate_kvnd_dist=_parse_dist(cogs_raw["partner_court_hourly_rate_kvnd"]),
        membership_event_partner_court_per_member_kvnd_dist=_parse_dist(
            cogs_raw["membership_event_and_partner_court_cogs_per_member_kvnd"]
        ),
    )

    f_opex = opex_raw["facility_opex"]
    opex = OpExConfig(
        venue_lease_dist=_parse_dist(f_opex["venue_lease_monthly_mvnd"]),
        electricity_lighting_water_dist=_parse_dist(f_opex["electricity_lighting_water_monthly_mvnd"]),
        maintenance_and_insurance_dist=_parse_dist(f_opex["maintenance_and_insurance_monthly_mvnd"]),
    )

    sga_raw = opex_raw["sga"]
    sga = SGAConfig(
        staff_wages_dist=_parse_dist(sga_raw["staff_wages_monthly_mvnd"]),
        booking_system_and_admin_dist=_parse_dist(sga_raw["booking_system_and_admin_monthly_mvnd"]),
        marketing_dist=_parse_dist(sga_raw["marketing_monthly_mvnd"]),
    )

    v2_raw = opex_raw["venue2_opex_and_sga"]
    venue2_opex_sga = Venue2OpExSGAConfig(
        lease_start_month_index=int(v2_raw["lease_start_month_index"]),
        operations_open_month_index=int(v2_raw["operations_open_month_index"]),
        venue_lease_dist=_parse_dist(v2_raw["venue_lease_monthly_mvnd"]),
        electricity_lighting_water_dist=_parse_dist(v2_raw["electricity_lighting_water_monthly_mvnd"]),
        maintenance_and_insurance_dist=_parse_dist(v2_raw["maintenance_and_insurance_monthly_mvnd"]),
        staff_wages_dist=_parse_dist(v2_raw["staff_wages_monthly_mvnd"]),
    )

    pm_raw = opex_raw["platform_monthly_opex"]
    pm_hdr = opex_raw.get("platform_Monthly_opex", pm_raw)
    platform_opex = PlatformMonthlyOpExConfig(
        activation_month_index=int(pm_hdr.get("activation_month_index", 5)),
        post_hypercare_month_index=int(pm_hdr.get("post_hypercare_month_index", 7)),
        cloud_hosting_db_redis_dist=_parse_dist(pm_raw["cloud_hosting_db_redis_mvnd"]),
        zalo_zns_otp_notifications_dist=_parse_dist(pm_raw["zalo_zns_otp_notifications_mvnd"]),
        observability_cicd_security_saas_dist=_parse_dist(pm_raw["observability_cicd_security_saas_mvnd"]),
        ongoing_maintenance_patching_retainer_dist=_parse_dist(pm_raw["ongoing_maintenance_patching_retainer_mvnd"]),
        school_university_community_ops_dist=_parse_dist(pm_raw["school_university_community_ops_mvnd"]),
        task_buffer_utilization_dist=_parse_dist(opex_raw["platform_cost_rates"]["task_buffer_utilization_dist"]),
    )

    vs_raw = capex_raw["venue_scaling"]
    venue_scaling = VenueScalingCapExConfig(
        enabled=bool(vs_raw["enabled"]),
        existing_courts=int(vs_raw["existing_courts"]),
        new_venue_courts=int(vs_raw["new_venue_courts"]),
        total_expanded_courts=int(vs_raw["total_expanded_courts"]),
        capacity_ratio_new_vs_existing=float(vs_raw["capacity_ratio_new_vs_existing"]),
        build_start_month_index=int(vs_raw["build_start_month_index"]),
        build_duration_months=int(vs_raw["build_duration_months"]),
        build_end_month_index=int(vs_raw["build_end_month_index"]),
        open_month_index=int(vs_raw["open_month_index"]),
        total_fitout_capex_mvnd=float(vs_raw["total_fitout_capex_mvnd"]),
        even_monthly_fitout_capex_mvnd=float(vs_raw["even_monthly_fitout_capex_mvnd"]),
        investor_total_new_money_mvnd=float(vs_raw["investor_total_new_money_mvnd"]),
        investor_reserve_mvnd=float(vs_raw["investor_reserve_mvnd"]),
        investor_signing_tranche_month_index=int(vs_raw.get("investor_signing_tranche_month_index", 1)),
        investor_signing_tranche_mvnd=float(vs_raw.get("investor_signing_tranche_mvnd", 500.0)),
        investor_equity_injection_month_index=int(vs_raw["investor_equity_injection_month_index"]),
        investor_expansion_tranche_mvnd=float(vs_raw.get("investor_expansion_tranche_mvnd", 7500.0)),
        new_venue_lease_monthly_mvnd=float(vs_raw["new_venue_lease_monthly_mvnd"]),
        lease_start_month_index=int(vs_raw["lease_start_month_index"]),
        fill_up_duration_months_dist=_parse_dist(vs_raw["fill_up_duration_months_dist"]),
        initial_open_utilization_fraction_dist=_parse_dist(vs_raw["initial_open_utilization_fraction_dist"]),
        network_customer_catchment_expansion_dist=_parse_dist(vs_raw["network_customer_catchment_expansion_dist"]),
    )

    pt_raw = ext_raw["pickleball_trend"]
    ln_raw = ext_raw["legal_and_noise_handling"]
    cp_raw = ext_raw["competitor_pressure"]
    wd_raw = ext_raw["weather_disruption"]
    ic_raw = ext_raw["input_cost_inflation"]
    u18_raw = ext_raw["under18_safety_compliance"]
    externals = ExternalsConfig(
        initial_trend_index=float(pt_raw["initial_trend_index"]),
        min_cumulative_trend_floor=float(pt_raw["min_cumulative_trend_floor"]),
        max_cumulative_trend_ceiling=float(pt_raw["max_cumulative_trend_ceiling"]),
        community_retention_insulation=float(pt_raw["community_retention_insulation"]),
        monthly_interest_change_rate_dist=_parse_dist(pt_raw["monthly_interest_change_rate"]),
        legal_and_noise_monthly_cost_dist=_parse_dist(ln_raw["monthly_cost_mvnd"]),
        legal_venue2_scaling_multiplier_dist=_parse_dist(ln_raw["venue2_scaling_multiplier"]),
        competitor_non_member_drag_rate_dist=_parse_dist(cp_raw["non_member_utilization_drag_rate"]),
        weather_utilization_multiplier_dist=_parse_dist(wd_raw["monthly_utilization_multiplier"]),
        input_cost_drift_rate_dist=_parse_dist(ic_raw["monthly_cost_drift_rate"]),
        under18_safety_monthly_cost_dist=_parse_dist(u18_raw["monthly_cost_mvnd"]),
    )

    tf_raw = capex_raw["taxes_and_financing"]
    nc_raw = capex_raw["new_capex"]
    tax = TaxesAndOtherConfig(
        corporate_tax_rate=float(tf_raw["corporate_tax_rate"]),
        tax_consolidation_month=int(tf_raw.get("tax_consolidation_month", 4)),
        vat_output_rate=float(tf_raw.get("vat_output_rate", 0.08)),
        vat_input_rate=float(tf_raw.get("vat_input_rate", 0.10)),
        venue2_depreciation_useful_life_months=int(tf_raw.get("venue2_depreciation_useful_life_months", 60)),
        depreciation_amortization_mvnd=float(nc_raw["depreciation_amortization_monthly_mvnd"]["mean"]),
        interest_expense_mvnd=float(tf_raw["interest_expense_monthly_mvnd"]),
    )

    bs_raw = capex_raw["balance_sheet_initial"]
    bs = BalanceSheetConfig(
        starting_cash_mvnd=float(bs_raw["starting_cash_mvnd"]),
        inventory_mvnd=float(bs_raw["inventory_mvnd"]),
        accounts_receivable_mvnd=float(bs_raw["accounts_receivable_mvnd"]),
        prepaid_expenses_mvnd=float(bs_raw["prepaid_expenses_mvnd"]),
        equipment_ppe_gross_mvnd=float(bs_raw["equipment_ppe_gross_mvnd"]),
        intangible_assets_mvnd=float(bs_raw["intangible_assets_mvnd"]),
        accounts_payable_mvnd=float(bs_raw["accounts_payable_mvnd"]),
        accrued_expenses_payable_mvnd=float(bs_raw["accrued_expenses_payable_mvnd"]),
        short_term_debt_mvnd=float(bs_raw["short_term_debt_mvnd"]),
        long_term_debt_mvnd=float(bs_raw["long_term_debt_mvnd"]),
        contributed_capital_mvnd=float(bs_raw["contributed_capital_mvnd"]),
        starting_retained_earnings_mvnd=float(bs_raw["starting_retained_earnings_mvnd"]),
        monthly_capex_mvnd=float(nc_raw["monthly_capex_mvnd"]["mean"]),
        monthly_equity_funding_mvnd=float(bs_raw["monthly_equity_funding_mvnd"]),
        monthly_dividends_mvnd=float(bs_raw["monthly_dividends_mvnd"]),
    )

    mc_raw = rand_raw["monte_carlo"]
    st_raw = rand_raw["stress_test_scenario"]
    randomness = MonteCarloRandomnessConfig(
        random_seed=int(mc_raw["random_seed"]),
        n_simulations=int(mc_raw["n_simulations"]),
        train_fraction=float(mc_raw["train_fraction"]),
        ci_low_percentile=float(mc_raw["ci_low_percentile"]),
        ci_high_percentile=float(mc_raw["ci_high_percentile"]),
        monthly_demand_macro_shock_dist=_parse_dist(rand_raw["monthly_demand_macro_shock"]),
        subscriber_acquisition_shock_dist=_parse_dist(rand_raw["subscriber_acquisition_random_shock"]),
        subscriber_churn_shock_dist=_parse_dist(rand_raw["subscriber_churn_random_shock"]),
        ancillary_revenue_shock_dist=_parse_dist(rand_raw["ancillary_revenue_random_shock"]),
        payment_gateway_cash_mix_shock_dist=_parse_dist(rand_raw["payment_gateway_cash_mix_shock"]),
        stress_demand_drop_fraction=float(st_raw["demand_drop_fraction"]),
        stress_demand_multiplier=float(st_raw["demand_multiplier"]),
    )

    return FinancialModelConfig(
        config_dir=config_dir,
        company=comp,
        operations=ops,
        customers=cust,
        revenue=rev,
        cogs=cogs,
        opex=opex,
        sga=sga,
        venue2_opex_sga=venue2_opex_sga,
        platform_opex=platform_opex,
        venue_scaling=venue_scaling,
        externals=externals,
        taxes_and_other=tax,
        balance_sheet=bs,
        randomness=randomness,
    )
