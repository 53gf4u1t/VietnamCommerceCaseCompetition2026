"""Machine learning adaptive pricing model and rigorous evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    median_absolute_error,
)

from .config import AppConfig
from .simulator import SimulationResults


FEATURE_NAMES = [
    "Court 1 State (0/1)",
    "Court 2 State (0/1)",
    "Court 3 State (0/1)",
    "Court 4 State (0/1)",
    "Court 5 State (0/1)",
    "Court 6 State (0/1)",
    "Hour of Day (6-21)",
    "Day of Week (0-6)",
    "Day of Month (1-30)",
]


@dataclass
class MLTechnicalMetrics:
    """Comprehensive ML regression, generalization, and economic optimality metrics."""

    n_train_samples: int
    n_test_samples: int

    # In-sample (Training Set: 240 months = 115,200 court-hours)
    train_r2: float
    train_rmse_kvnd: float
    train_mae_kvnd: float
    train_mape_pct: float
    train_explained_var: float

    # Out-of-sample (Held-out Test Set: 60 months = 28,800 court-hours)
    test_r2: float
    test_rmse_kvnd: float
    test_mae_kvnd: float
    test_medae_kvnd: float
    test_mape_pct: float
    test_explained_var: float
    test_within_15k_acc_pct: float
    test_within_25k_acc_pct: float

    # Regime-specific Test Metrics (WD Daytime, WD Evening, Weekend)
    regime_r2: dict[str, float]
    regime_rmse_kvnd: dict[str, float]
    regime_mae_kvnd: dict[str, float]
    regime_mape_pct: dict[str, float]
    regime_optimality_capture_pct: dict[str, float]

    # Economic Regret & Oracle Optimality Capture
    oracle_capture_ratio_pct: float
    uplift_capture_vs_oracle_pct: float
    mean_hourly_regret_kvnd: float
    median_hourly_regret_kvnd: float
    p90_hourly_regret_kvnd: float

    # Feature Importances (Permutation Importance on Test Set)
    feature_names: list[str]
    perm_importance_mean: np.ndarray
    perm_importance_std: np.ndarray

    # Iterative Learning Curves (Train vs Test RMSE across boosting iterations)
    iter_steps: np.ndarray
    iter_train_rmse: np.ndarray
    iter_test_rmse: np.ndarray

    # Raw Test Arrays for Diagnostic Plotting
    y_test_true: np.ndarray
    y_test_pred_raw: np.ndarray
    y_test_pred_rounded: np.ndarray
    test_residuals: np.ndarray
    test_regime_flat: np.ndarray


@dataclass
class EvaluationResults:
    """Out-of-sample evaluation metrics and representative monthly calendar matrices."""

    ml_metrics: MLTechnicalMetrics

    # Array of shape (n_test_sims, days_per_month, hours_per_day)
    ml_prices_test: np.ndarray
    ml_expected_demand_test: np.ndarray
    ml_realized_demand_test: np.ndarray
    ml_revenue_test_kvnd: np.ndarray
    baseline_revenue_test_kvnd: np.ndarray
    baseline_demand_test: np.ndarray

    # Daily revenue aggregated across test simulations: shape (days_per_month,)
    daily_baseline_rev_mean_kvnd: np.ndarray
    daily_baseline_rev_std_kvnd: np.ndarray
    daily_ml_rev_mean_kvnd: np.ndarray
    daily_ml_rev_std_kvnd: np.ndarray

    # Single representative test simulation trajectory (Day 1..30) for realistic daily variation visualization
    rep_daily_baseline_rev_kvnd: np.ndarray
    rep_daily_ml_rev_kvnd: np.ndarray

    # Representative month (30 days x 16 hours) for calendar heatmap visualization
    rep_month_ml_price: np.ndarray
    rep_month_base_price: np.ndarray
    rep_month_ml_util: np.ndarray
    rep_month_base_util: np.ndarray

    # Monte Carlo test-set mean calendar matrices (30 days x 16 hours) for downstream financial modeling
    mc_month_ml_price: np.ndarray
    mc_month_ml_util: np.ndarray
    mc_month_base_util: np.ndarray

    # Summary KPIs across out-of-sample test months
    mean_baseline_monthly_rev_kvnd: float
    std_baseline_monthly_rev_kvnd: float
    mean_ml_monthly_rev_kvnd: float
    std_ml_monthly_rev_kvnd: float
    mean_oracle_monthly_rev_kvnd: float

    mean_baseline_booked_hours: float
    mean_ml_booked_hours: float

    baseline_wd_daytime_util: float
    ml_wd_daytime_util: float
    baseline_wd_evening_util: float
    ml_wd_evening_util: float
    baseline_weekend_util: float
    ml_weekend_util: float

    # 30% lower demand stress-test KPIs (Case requirement on Page 4)
    stress30_baseline_monthly_rev_kvnd: float
    stress30_ml_monthly_rev_kvnd: float
    stress30_baseline_wd_daytime_util: float
    stress30_ml_wd_daytime_util: float


def _build_feature_matrix(
    court_states: np.ndarray,
    hours: np.ndarray,
) -> np.ndarray:
    """
    Build the exact 9-feature input vector requested by the case specification:
      [c_1, c_2, c_3, c_4, c_5, c_6, hour_of_day, day_of_week, day_of_month]
    for every (sim, day, hour) observation.
    """
    n_sims, n_days, n_hours, n_courts = court_states.shape
    total_rows = n_sims * n_days * n_hours
    X = np.empty((total_rows, n_courts + 3), dtype=np.float64)
    X[:, :n_courts] = court_states.reshape(total_rows, n_courts)

    d_idx = np.arange(n_days, dtype=np.float64)
    dow_grid = np.broadcast_to((d_idx % 7)[None, :, None], (n_sims, n_days, n_hours))
    dom_grid = np.broadcast_to((d_idx + 1.0)[None, :, None], (n_sims, n_days, n_hours))
    h_grid = np.broadcast_to(hours.astype(np.float64)[None, None, :], (n_sims, n_days, n_hours))

    X[:, n_courts] = h_grid.reshape(-1)
    X[:, n_courts + 1] = dow_grid.reshape(-1)
    X[:, n_courts + 2] = dom_grid.reshape(-1)
    return X


def train_and_evaluate(
    cfg: AppConfig,
    sim: SimulationResults,
) -> tuple[HistGradientBoostingRegressor, EvaluationResults]:
    """
    Train the ML pricing model on simulated training trajectories to predict the
    revenue-maximizing court-hour price from the 6 binary court rental states and
    temporal features, then compute rigorous technical & economic metrics on held-out test months.
    """
    n_sims = cfg.simulation.n_simulations
    n_days = cfg.simulation.days_per_month
    n_hours = cfg.simulation.hours_per_day
    n_courts = cfg.simulation.courts_count
    hours = cfg.simulation.hours_array
    bounds = cfg.pricing_bounds

    n_train_sims = int(round(n_sims * cfg.simulation.train_fraction))
    n_test_sims = n_sims - n_train_sims

    X_all = _build_feature_matrix(sim.court_states, hours)
    y_all = sim.optimal_price.reshape(-1)

    split_row = n_train_sims * n_days * n_hours
    X_train, y_train = X_all[:split_row], y_all[:split_row]
    X_test, y_test = X_all[split_row:], y_all[split_row:]

    model = HistGradientBoostingRegressor(
        max_iter=cfg.ml_model.max_iter,
        max_depth=cfg.ml_model.max_depth,
        learning_rate=cfg.ml_model.learning_rate,
        min_samples_leaf=cfg.ml_model.min_samples_leaf,
        l2_regularization=cfg.ml_model.l2_regularization,
        early_stopping=False,
        random_state=cfg.simulation.random_seed,
    )
    model.fit(X_train, y_train)

    y_train_pred = np.clip(model.predict(X_train), bounds.min_price_kvnd, bounds.max_price_kvnd)
    y_test_pred_raw = np.clip(model.predict(X_test), bounds.min_price_kvnd, bounds.max_price_kvnd)

    raw_preds_3d = y_test_pred_raw.reshape(n_test_sims, n_days, n_hours)

    # Apply transparent pricing guardrails and round to nearest 5k VND step
    step = bounds.price_step_kvnd
    ml_prices_test = np.round(raw_preds_3d / step) * step
    ml_prices_test = np.clip(
        ml_prices_test,
        bounds.min_price_kvnd,
        bounds.max_price_kvnd,
    )
    late_idx = bounds.late_night_hour - cfg.simulation.open_hour
    if 0 <= late_idx < n_hours:
        ml_prices_test[:, :, late_idx] = np.minimum(
            ml_prices_test[:, :, late_idx],
            bounds.late_night_max_price_kvnd,
        )

    y_test_pred_rounded = ml_prices_test.reshape(-1)
    test_residuals = y_test - y_test_pred_raw

    # Compute Staged Learning Curve across boosting iterations
    n_iters = model.n_iter_
    step_indices = np.unique(np.linspace(1, n_iters, min(20, n_iters), dtype=int))
    step_set = set(int(x) for x in step_indices)
    iter_train_rmse = []
    iter_test_rmse = []

    X_tr_sub, y_tr_sub = X_train[::4], y_train[::4]
    train_staged_gen = model.staged_predict(X_tr_sub)
    test_staged_gen = model.staged_predict(X_test)
    for i, (tr_p, te_p) in enumerate(zip(train_staged_gen, test_staged_gen), start=1):
        if i in step_set:
            iter_train_rmse.append(float(root_mean_squared_error(y_tr_sub, tr_p)))
            iter_test_rmse.append(float(root_mean_squared_error(y_test, te_p)))

    # Permutation Importance on Out-of-Sample Test Set
    perm_res = permutation_importance(
        model,
        X_test[::2],
        y_test[::2],
        n_repeats=3,
        random_state=cfg.simulation.random_seed,
        scoring="r2",
    )

    # Extract test slice parameters
    a_test = sim.slope_a[n_train_sims:]
    b_test = sim.intercept_b[n_train_sims:]
    base_p_test = sim.baseline_price[n_train_sims:]
    opt_p_test = sim.optimal_price[n_train_sims:]
    reg_test = sim.regime_id[n_train_sims:]
    base_states_test = sim.court_states[n_train_sims:]

    baseline_demand_test = base_states_test.sum(axis=-1).astype(np.float64)
    baseline_revenue_test_kvnd = baseline_demand_test * base_p_test

    # Evaluate ML counterfactual demand along the linear demand curve Q(x) = clip(a*x + b, 0, 6)
    ml_expected_demand_test = np.clip(
        a_test * ml_prices_test + b_test,
        0.0,
        float(n_courts),
    )

    rng = np.random.default_rng(cfg.simulation.random_seed + 999)
    ml_prob_test = np.clip(ml_expected_demand_test / float(n_courts), 0.0, 1.0)
    ml_realized_demand_test = rng.binomial(n=n_courts, p=ml_prob_test).astype(np.float64)

    ml_revenue_test_kvnd = ml_expected_demand_test * ml_prices_test

    # Oracle (perfect foresight of a, b) revenue for upper-bound benchmarking
    oracle_demand_test = np.clip(a_test * opt_p_test + b_test, 0.0, float(n_courts))
    oracle_revenue_test_kvnd = oracle_demand_test * opt_p_test

    hourly_regret = np.maximum(0.0, oracle_revenue_test_kvnd - ml_revenue_test_kvnd).reshape(-1)

    # Regime-specific technical breakdown
    reg_flat = reg_test.reshape(-1)
    regime_names = {
        0: "Weekday Daytime (06:00-17:00)",
        1: "Weekday Evening (17:00-22:00)",
        2: "Weekend (Sat-Sun 06:00-22:00)",
    }
    regime_r2 = {}
    regime_rmse = {}
    regime_mae = {}
    regime_mape = {}
    regime_opt_cap = {}

    for r_id, r_name in regime_names.items():
        m_flat = reg_flat == r_id
        m_3d = reg_test == r_id
        regime_r2[r_name] = float(r2_score(y_test[m_flat], y_test_pred_raw[m_flat]))
        regime_rmse[r_name] = float(root_mean_squared_error(y_test[m_flat], y_test_pred_raw[m_flat]))
        regime_mae[r_name] = float(mean_absolute_error(y_test[m_flat], y_test_pred_raw[m_flat]))
        regime_mape[r_name] = float(mean_absolute_percentage_error(y_test[m_flat], y_test_pred_raw[m_flat]) * 100.0)
        regime_opt_cap[r_name] = float(
            (ml_revenue_test_kvnd[m_3d].sum() / oracle_revenue_test_kvnd[m_3d].sum()) * 100.0
        )

    # Daily revenue summaries
    daily_base_per_sim = baseline_revenue_test_kvnd.sum(axis=-1)
    daily_ml_per_sim = ml_revenue_test_kvnd.sum(axis=-1)

    daily_baseline_rev_mean_kvnd = daily_base_per_sim.mean(axis=0)
    daily_baseline_rev_std_kvnd = daily_base_per_sim.std(axis=0)
    daily_ml_rev_mean_kvnd = daily_ml_per_sim.mean(axis=0)
    daily_ml_rev_std_kvnd = daily_ml_per_sim.std(axis=0)

    monthly_base_totals = daily_base_per_sim.sum(axis=-1)
    monthly_ml_totals = daily_ml_per_sim.sum(axis=-1)
    monthly_oracle_totals = oracle_revenue_test_kvnd.sum(axis=(-1, -2))

    mean_base_rev = float(monthly_base_totals.mean())
    mean_ml_rev = float(monthly_ml_totals.mean())
    mean_oracle_rev = float(monthly_oracle_totals.mean())

    oracle_capture_ratio_pct = (mean_ml_rev / mean_oracle_rev) * 100.0
    uplift_capture_vs_oracle_pct = (
        (mean_ml_rev - mean_base_rev) / max(1e-6, (mean_oracle_rev - mean_base_rev))
    ) * 100.0

    ml_metrics = MLTechnicalMetrics(
        n_train_samples=int(len(X_train)),
        n_test_samples=int(len(X_test)),
        train_r2=float(r2_score(y_train, y_train_pred)),
        train_rmse_kvnd=float(root_mean_squared_error(y_train, y_train_pred)),
        train_mae_kvnd=float(mean_absolute_error(y_train, y_train_pred)),
        train_mape_pct=float(mean_absolute_percentage_error(y_train, y_train_pred) * 100.0),
        train_explained_var=float(explained_variance_score(y_train, y_train_pred)),
        test_r2=float(r2_score(y_test, y_test_pred_raw)),
        test_rmse_kvnd=float(root_mean_squared_error(y_test, y_test_pred_raw)),
        test_mae_kvnd=float(mean_absolute_error(y_test, y_test_pred_raw)),
        test_medae_kvnd=float(median_absolute_error(y_test, y_test_pred_raw)),
        test_mape_pct=float(mean_absolute_percentage_error(y_test, y_test_pred_raw) * 100.0),
        test_explained_var=float(explained_variance_score(y_test, y_test_pred_raw)),
        test_within_15k_acc_pct=float((np.abs(test_residuals) <= 15.0).mean() * 100.0),
        test_within_25k_acc_pct=float((np.abs(test_residuals) <= 25.0).mean() * 100.0),
        regime_r2=regime_r2,
        regime_rmse_kvnd=regime_rmse,
        regime_mae_kvnd=regime_mae,
        regime_mape_pct=regime_mape,
        regime_optimality_capture_pct=regime_opt_cap,
        oracle_capture_ratio_pct=oracle_capture_ratio_pct,
        uplift_capture_vs_oracle_pct=uplift_capture_vs_oracle_pct,
        mean_hourly_regret_kvnd=float(hourly_regret.mean()),
        median_hourly_regret_kvnd=float(np.median(hourly_regret)),
        p90_hourly_regret_kvnd=float(np.percentile(hourly_regret, 90)),
        feature_names=FEATURE_NAMES,
        perm_importance_mean=perm_res.importances_mean,
        perm_importance_std=perm_res.importances_std,
        iter_steps=step_indices,
        iter_train_rmse=np.array(iter_train_rmse),
        iter_test_rmse=np.array(iter_test_rmse),
        y_test_true=y_test,
        y_test_pred_raw=y_test_pred_raw,
        y_test_pred_rounded=y_test_pred_rounded,
        test_residuals=test_residuals,
        test_regime_flat=reg_flat,
    )

    # Representative month calendar matrices (select the test trajectory closest to median monthly revenue)
    rep_idx = int(np.argmin(np.abs(monthly_base_totals - np.median(monthly_base_totals))))
    rep_daily_baseline_rev_kvnd = daily_base_per_sim[rep_idx]
    rep_daily_ml_rev_kvnd = (ml_realized_demand_test[rep_idx] * ml_prices_test[rep_idx]).sum(axis=-1)

    rep_month_ml_price = ml_prices_test[rep_idx]
    rep_month_base_price = base_p_test[rep_idx]
    rep_month_ml_util = ml_expected_demand_test[rep_idx] / float(n_courts)
    rep_month_base_util = baseline_demand_test[rep_idx] / float(n_courts)

    mc_month_ml_price = ml_prices_test.mean(axis=0)
    mc_month_ml_util = ml_expected_demand_test.mean(axis=0) / float(n_courts)
    mc_month_base_util = baseline_demand_test.mean(axis=0) / float(n_courts)

    wd_day_mask = reg_test == 0
    wd_eve_mask = reg_test == 1
    wknd_mask = reg_test == 2

    base_wd_day_u = float(baseline_demand_test[wd_day_mask].mean() / n_courts)
    ml_wd_day_u = float(ml_expected_demand_test[wd_day_mask].mean() / n_courts)

    base_wd_eve_u = float(baseline_demand_test[wd_eve_mask].mean() / n_courts)
    ml_wd_eve_u = float(ml_expected_demand_test[wd_eve_mask].mean() / n_courts)

    base_wknd_u = float(baseline_demand_test[wknd_mask].mean() / n_courts)
    ml_wknd_u = float(ml_expected_demand_test[wknd_mask].mean() / n_courts)

    # 30% lower demand stress-test scenario
    q_base_shock = np.clip(0.70 * (a_test * base_p_test + b_test), 0.0, float(n_courts))
    rev_base_shock = (q_base_shock * base_p_test).sum(axis=(-1, -2)).mean()

    ml_shock_prices = np.clip(
        np.round((ml_prices_test * 0.88) / step) * step,
        bounds.min_price_kvnd,
        bounds.max_price_kvnd,
    )
    q_ml_shock = np.clip(0.70 * (a_test * ml_shock_prices + b_test), 0.0, float(n_courts))
    rev_ml_shock = (q_ml_shock * ml_shock_prices).sum(axis=(-1, -2)).mean()

    stress30_base_wd_day_u = float(q_base_shock[wd_day_mask].mean() / n_courts)
    stress30_ml_wd_day_u = float(q_ml_shock[wd_day_mask].mean() / n_courts)

    return model, EvaluationResults(
        ml_metrics=ml_metrics,
        ml_prices_test=ml_prices_test,
        ml_expected_demand_test=ml_expected_demand_test,
        ml_realized_demand_test=ml_realized_demand_test,
        ml_revenue_test_kvnd=ml_revenue_test_kvnd,
        baseline_revenue_test_kvnd=baseline_revenue_test_kvnd,
        baseline_demand_test=baseline_demand_test,
        daily_baseline_rev_mean_kvnd=daily_baseline_rev_mean_kvnd,
        daily_baseline_rev_std_kvnd=daily_baseline_rev_std_kvnd,
        daily_ml_rev_mean_kvnd=daily_ml_rev_mean_kvnd,
        daily_ml_rev_std_kvnd=daily_ml_rev_std_kvnd,
        rep_daily_baseline_rev_kvnd=rep_daily_baseline_rev_kvnd,
        rep_daily_ml_rev_kvnd=rep_daily_ml_rev_kvnd,
        rep_month_ml_price=rep_month_ml_price,
        rep_month_base_price=rep_month_base_price,
        rep_month_ml_util=rep_month_ml_util,
        rep_month_base_util=rep_month_base_util,
        mc_month_ml_price=mc_month_ml_price,
        mc_month_ml_util=mc_month_ml_util,
        mc_month_base_util=mc_month_base_util,
        mean_baseline_monthly_rev_kvnd=mean_base_rev,
        std_baseline_monthly_rev_kvnd=float(monthly_base_totals.std()),
        mean_ml_monthly_rev_kvnd=mean_ml_rev,
        std_ml_monthly_rev_kvnd=float(monthly_ml_totals.std()),
        mean_oracle_monthly_rev_kvnd=mean_oracle_rev,
        mean_baseline_booked_hours=float(baseline_demand_test.sum(axis=(-1, -2)).mean()),
        mean_ml_booked_hours=float(ml_expected_demand_test.sum(axis=(-1, -2)).mean()),
        baseline_wd_daytime_util=base_wd_day_u,
        ml_wd_daytime_util=ml_wd_day_u,
        baseline_wd_evening_util=base_wd_eve_u,
        ml_wd_evening_util=ml_wd_eve_u,
        baseline_weekend_util=base_wknd_u,
        ml_weekend_util=ml_wknd_u,
        stress30_baseline_monthly_rev_kvnd=float(rev_base_shock),
        stress30_ml_monthly_rev_kvnd=float(rev_ml_shock),
        stress30_baseline_wd_daytime_util=stress30_base_wd_day_u,
        stress30_ml_wd_daytime_util=stress30_ml_wd_day_u,
    )
