"""Monte Carlo simulation engine for court rental states and linear demand curves."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import AppConfig


@dataclass
class SimulationResults:
    """Container for all simulated monthly trajectories and court-hour observations."""

    # Array of shape (n_sims, days_per_month, hours_per_day, courts_count) with binary 0/1 states
    court_states: np.ndarray
    # Array of shape (n_sims, days_per_month, hours_per_day) with linear demand slope `a` (< 0)
    slope_a: np.ndarray
    # Array of shape (n_sims, days_per_month, hours_per_day) with linear demand intercept `b` (> 0)
    intercept_b: np.ndarray
    # Array of shape (n_sims, days_per_month, hours_per_day) with baseline static price (160 or 260 k VND)
    baseline_price: np.ndarray
    # Array of shape (n_sims, days_per_month, hours_per_day) with true optimal price on [min_p, max_p]
    optimal_price: np.ndarray
    # Array of shape (n_sims, days_per_month, hours_per_day) with regime code (0=WD Day, 1=WD Eve, 2=Weekend)
    regime_id: np.ndarray

    # Sampled daily multipliers (n_sims, days_per_month) & hourly shocks (n_sims, days_per_month, hours_per_day)
    sampled_daily_multipliers: np.ndarray
    sampled_hourly_shocks: np.ndarray

    # Monthly summary arrays of shape (n_sims,)
    monthly_booked_hours: np.ndarray
    monthly_wd_daytime_util: np.ndarray
    monthly_wd_evening_util: np.ndarray
    monthly_weekend_util: np.ndarray
    monthly_baseline_revenue_kvnd: np.ndarray


def compute_optimal_price(
    a: float,
    b: float,
    courts_count: int,
    min_price: float,
    max_price: float,
) -> float:
    """
    Compute the revenue-maximizing price x* for linear demand Q(x) = clip(a*x + b, 0, courts_count).

    For x <= x_cap = (courts_count - b) / a, demand is capped at `courts_count`, so revenue
    is `courts_count * x`, which strictly increases up to `x_cap`.
    For x >= x_cap, unconstrained quadratic revenue `x * (a*x + b)` has vertex at `x_vertex = -b / (2*a)`.
    Therefore, the optimal unconstrained price on the capped linear curve is `max(x_vertex, x_cap)`.
    """
    x_vertex = -b / (2.0 * a)
    x_cap = (float(courts_count) - b) / a
    return float(np.clip(max(x_vertex, x_cap), min_price, max_price))


def run_simulations(cfg: AppConfig) -> SimulationResults:
    """
    Run Monte Carlo simulations of court rental states and demand curves over N months.
    All variations are sampled purely from configured probability distributions without
    hardcoded day-of-week arrays, sine waves, or preset hourly lookup profiles.
    """
    rng = np.random.default_rng(cfg.simulation.random_seed)

    n_sims = cfg.simulation.n_simulations
    n_days = cfg.simulation.days_per_month
    n_hours = cfg.simulation.hours_per_day
    n_courts = cfg.simulation.courts_count
    hours = cfg.simulation.hours_array

    court_states = np.zeros((n_sims, n_days, n_hours, n_courts), dtype=np.int8)
    slope_a = np.zeros((n_sims, n_days, n_hours), dtype=np.float64)
    intercept_b = np.zeros((n_sims, n_days, n_hours), dtype=np.float64)
    baseline_price = np.zeros((n_sims, n_days, n_hours), dtype=np.float64)
    optimal_price = np.zeros((n_sims, n_days, n_hours), dtype=np.float64)
    regime_id = np.zeros((n_sims, n_days, n_hours), dtype=np.int8)

    sampled_daily_multipliers = np.zeros((n_sims, n_days), dtype=np.float64)
    sampled_hourly_shocks = np.zeros((n_sims, n_days, n_hours), dtype=np.float64)

    monthly_booked_hours = np.zeros(n_sims, dtype=np.float64)
    monthly_wd_daytime_util = np.zeros(n_sims, dtype=np.float64)
    monthly_wd_evening_util = np.zeros(n_sims, dtype=np.float64)
    monthly_weekend_util = np.zeros(n_sims, dtype=np.float64)
    monthly_baseline_revenue_kvnd = np.zeros(n_sims, dtype=np.float64)

    dc = cfg.demand_curve
    bounds = cfg.pricing_bounds

    for s in range(n_sims):
        # 1. Sample month-level regime baseline utilizations from configured Normal distributions
        u_wd_day_s = float(
            np.clip(
                rng.normal(cfg.weekday_daytime.mean, cfg.weekday_daytime.std),
                cfg.weekday_daytime.min,
                cfg.weekday_daytime.max,
            )
        )
        u_wd_eve_s = float(
            np.clip(
                rng.normal(cfg.weekday_evening.mean, cfg.weekday_evening.std),
                cfg.weekday_evening.min,
                cfg.weekday_evening.max,
            )
        )
        u_wknd_s = float(
            np.clip(
                rng.normal(cfg.weekend.mean, cfg.weekend.std),
                cfg.weekend.min,
                cfg.weekend.max,
            )
        )

        wd_day_booked = 0
        wd_eve_booked = 0
        wknd_booked = 0
        total_rev = 0.0

        for d in range(n_days):
            dow = d % 7  # 0..4 = Weekday (22 days/mo), 5..6 = Weekend (8 days/mo)
            is_weekend = dow >= 5

            # 2. Sample a purely random day multiplier from the configured daily_random_multiplier distribution
            day_mult = cfg.daily_random_multiplier.sample(rng)
            sampled_daily_multipliers[s, d] = day_mult

            for h_idx, h in enumerate(hours):
                # 3. Sample a purely random hourly shock from the configured hourly_random_shock distribution
                hr_shock = cfg.hourly_random_shock.sample(rng)
                sampled_hourly_shocks[s, d, h_idx] = hr_shock

                if is_weekend:
                    reg = 2
                    p0 = cfg.baseline_pricing.peak_price_kvnd
                    # Apply random day multiplier & hourly shock around the 90% weekend anchor
                    # (damped near the 100% boundary so the mean stays anchored at 90%)
                    u_hr = float(
                        np.clip(
                            u_wknd_s * (1.0 + 0.25 * (day_mult * hr_shock - 1.0))
                            + rng.normal(0.0, cfg.weekend.hourly_noise_std),
                            cfg.weekend.min,
                            cfg.weekend.max,
                        )
                    )
                    a = float(
                        np.clip(
                            rng.normal(dc.weekend_a_mean, dc.weekend_a_std),
                            dc.a_min_clip,
                            dc.a_max_clip,
                        )
                    )
                elif h < 17:
                    reg = 0
                    p0 = cfg.baseline_pricing.off_peak_price_kvnd
                    # Apply random day multiplier & hourly shock around the 40% weekday daytime anchor
                    u_hr = float(
                        np.clip(
                            u_wd_day_s * day_mult * hr_shock
                            + rng.normal(0.0, cfg.weekday_daytime.hourly_noise_std),
                            cfg.weekday_daytime.min,
                            cfg.weekday_daytime.max,
                        )
                    )
                    a = float(
                        np.clip(
                            rng.normal(dc.weekday_daytime_a_mean, dc.weekday_daytime_a_std),
                            dc.a_min_clip,
                            dc.a_max_clip,
                        )
                    )
                else:
                    reg = 1
                    p0 = cfg.baseline_pricing.peak_price_kvnd
                    # Apply random day multiplier & hourly shock around the 95% weekday evening anchor
                    u_hr = float(
                        np.clip(
                            u_wd_eve_s * (1.0 + 0.15 * (day_mult * hr_shock - 1.0))
                            + rng.normal(0.0, cfg.weekday_evening.hourly_noise_std),
                            cfg.weekday_evening.min,
                            cfg.weekday_evening.max,
                        )
                    )
                    a = float(
                        np.clip(
                            rng.normal(dc.weekday_evening_a_mean, dc.weekday_evening_a_std),
                            dc.a_min_clip,
                            dc.a_max_clip,
                        )
                    )

                # Anchor the linear demand curve Q_r(x) = a*x + b at (p0, q0_latent)
                q0_latent = n_courts * u_hr
                b = q0_latent - a * p0

                # Sample binary 0/1 rental state for each of the 6 courts
                states = (rng.random(n_courts) < u_hr).astype(np.int8)
                k_booked = int(states.sum())

                # Compute optimal price given demand curve parameters (a, b) and guardrails
                max_p = (
                    bounds.late_night_max_price_kvnd
                    if h == bounds.late_night_hour
                    else bounds.max_price_kvnd
                )
                p_opt = compute_optimal_price(
                    a=a,
                    b=b,
                    courts_count=n_courts,
                    min_price=bounds.min_price_kvnd,
                    max_price=max_p,
                )

                court_states[s, d, h_idx] = states
                slope_a[s, d, h_idx] = a
                intercept_b[s, d, h_idx] = b
                baseline_price[s, d, h_idx] = p0
                optimal_price[s, d, h_idx] = p_opt
                regime_id[s, d, h_idx] = reg

                total_rev += k_booked * p0
                if reg == 0:
                    wd_day_booked += k_booked
                elif reg == 1:
                    wd_eve_booked += k_booked
                else:
                    wknd_booked += k_booked

        total_booked = wd_day_booked + wd_eve_booked + wknd_booked
        monthly_booked_hours[s] = total_booked
        monthly_wd_daytime_util[s] = wd_day_booked / (
            cfg.simulation.weekdays_per_month * 11 * n_courts
        )
        monthly_wd_evening_util[s] = wd_eve_booked / (
            cfg.simulation.weekdays_per_month * 5 * n_courts
        )
        monthly_weekend_util[s] = wknd_booked / (
            cfg.simulation.weekend_days_per_month * n_hours * n_courts
        )
        monthly_baseline_revenue_kvnd[s] = total_rev

    return SimulationResults(
        court_states=court_states,
        slope_a=slope_a,
        intercept_b=intercept_b,
        baseline_price=baseline_price,
        optimal_price=optimal_price,
        regime_id=regime_id,
        sampled_daily_multipliers=sampled_daily_multipliers,
        sampled_hourly_shocks=sampled_hourly_shocks,
        monthly_booked_hours=monthly_booked_hours,
        monthly_wd_daytime_util=monthly_wd_daytime_util,
        monthly_wd_evening_util=monthly_wd_evening_util,
        monthly_weekend_util=monthly_weekend_util,
        monthly_baseline_revenue_kvnd=monthly_baseline_revenue_kvnd,
    )
