"""Configuration loader and typed schema for PICK! Adaptive Pricing (reading from central CaseComp/configs/)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
import numpy as np


@dataclass(frozen=True)
class TruncatedNormalConfig:
    """Parameters for a truncated Normal(mean, std) distribution clipped to [min, max]."""

    mean: float
    std: float
    min: float
    max: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(np.clip(rng.normal(self.mean, self.std), self.min, self.max))


@dataclass(frozen=True)
class RegimeDistributionConfig:
    mean: float
    std: float
    min: float
    max: float
    hourly_noise_std: float


@dataclass(frozen=True)
class SimulationConfig:
    random_seed: int
    n_simulations: int
    train_fraction: float
    days_per_month: int
    weekdays_per_month: int
    weekend_days_per_month: int
    courts_count: int
    open_hour: int
    close_hour: int

    @property
    def hours_per_day(self) -> int:
        return self.close_hour - self.open_hour

    @property
    def hours_array(self) -> np.ndarray:
        return np.arange(self.open_hour, self.close_hour)


@dataclass(frozen=True)
class BaselinePricingConfig:
    off_peak_price_kvnd: float
    peak_price_kvnd: float
    target_monthly_booked_hours_mean: float
    target_monthly_booked_hours_std: float
    target_monthly_court_revenue_kvnd: float


@dataclass(frozen=True)
class DemandCurveConfig:
    weekday_daytime_a_mean: float
    weekday_daytime_a_std: float
    weekday_evening_a_mean: float
    weekday_evening_a_std: float
    weekend_a_mean: float
    weekend_a_std: float
    a_min_clip: float
    a_max_clip: float


@dataclass(frozen=True)
class PricingBoundsConfig:
    min_price_kvnd: float
    max_price_kvnd: float
    price_step_kvnd: float
    late_night_hour: int
    late_night_max_price_kvnd: float


@dataclass(frozen=True)
class AncillaryEconomicsConfig:
    paddle_balls_rev_per_court_hour_kvnd: float
    fnb_rev_per_court_hour_kvnd: float
    fnb_var_cost_ratio: float
    payment_consumables_var_ratio: float


@dataclass(frozen=True)
class MLModelConfig:
    max_iter: int
    max_depth: int
    learning_rate: float
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig
    daily_random_multiplier: TruncatedNormalConfig
    hourly_random_shock: TruncatedNormalConfig
    baseline_pricing: BaselinePricingConfig
    weekday_daytime: RegimeDistributionConfig
    weekday_evening: RegimeDistributionConfig
    weekend: RegimeDistributionConfig
    demand_curve: DemandCurveConfig
    pricing_bounds: PricingBoundsConfig
    ancillary: AncillaryEconomicsConfig
    ml_model: MLModelConfig


def _load_regime(data: dict) -> RegimeDistributionConfig:
    return RegimeDistributionConfig(
        mean=float(data["mean"]),
        std=float(data["std"]),
        min=float(data["min"]),
        max=float(data["max"]),
        hourly_noise_std=float(data["hourly_noise_std"]),
    )


def _load_trunc_norm(data: dict) -> TruncatedNormalConfig:
    return TruncatedNormalConfig(
        mean=float(data["mean"]),
        std=float(data["std"]),
        min=float(data["min"]),
        max=float(data["max"]),
    )


def load_config(configs_dir: Path) -> AppConfig:
    """Load and validate configuration from central CaseComp/configs/ directory."""
    configs_dir = configs_dir.resolve()
    if configs_dir.is_file():
        configs_dir = configs_dir.parent

    with (configs_dir / "pricing.toml").open("rb") as f:
        pricing_raw = tomllib.load(f)
    with (configs_dir / "randomness.toml").open("rb") as f:
        rand_raw = tomllib.load(f)

    mc = rand_raw["monte_carlo"]
    ops = pricing_raw["court_operations"]
    sim = SimulationConfig(
        random_seed=int(mc["random_seed"]),
        n_simulations=int(mc["n_simulations"]),
        train_fraction=float(mc["train_fraction"]),
        days_per_month=int(ops["days_per_month"]),
        weekdays_per_month=int(ops["weekdays_per_month"]),
        weekend_days_per_month=int(ops["weekend_days_per_month"]),
        courts_count=int(ops["courts_count"]),
        open_hour=int(ops["open_hour"]),
        close_hour=int(ops["close_hour"]),
    )

    daily_mult = _load_trunc_norm(rand_raw["daily_random_multiplier"])
    hourly_shock = _load_trunc_norm(rand_raw["hourly_random_shock"])
    base = BaselinePricingConfig(**pricing_raw["baseline_pricing"])

    ud = pricing_raw["utilization_distributions"]
    wd_day = _load_regime(ud["weekday_daytime"])
    wd_eve = _load_regime(ud["weekday_evening"])
    wknd = _load_regime(ud["weekend"])

    dc_raw = pricing_raw["demand_curve"]
    dc = DemandCurveConfig(
        weekday_daytime_a_mean=float(dc_raw["weekday_daytime_a_mean"]),
        weekday_daytime_a_std=float(dc_raw["weekday_daytime_a_std"]),
        weekday_evening_a_mean=float(dc_raw["weekday_evening_a_mean"]),
        weekday_evening_a_std=float(dc_raw["weekday_evening_a_std"]),
        weekend_a_mean=float(dc_raw["weekend_a_mean"]),
        weekend_a_std=float(dc_raw["weekend_a_std"]),
        a_min_clip=float(dc_raw["a_min_clip"]),
        a_max_clip=float(dc_raw["a_max_clip"]),
    )

    bounds = PricingBoundsConfig(**pricing_raw["pricing_bounds"])
    anc = AncillaryEconomicsConfig(**pricing_raw["ancillary_economics"])
    ml = MLModelConfig(**pricing_raw["ml_model"])

    return AppConfig(
        simulation=sim,
        daily_random_multiplier=daily_mult,
        hourly_random_shock=hourly_shock,
        baseline_pricing=base,
        weekday_daytime=wd_day,
        weekday_evening=wd_eve,
        weekend=wknd,
        demand_curve=dc,
        pricing_bounds=bounds,
        ancillary=anc,
        ml_model=ml,
    )
