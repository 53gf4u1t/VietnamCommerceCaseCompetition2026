"""Machine learning 4-segment marketing budget predictor (`HistGradientBoostingRegressor`) and evaluation suite."""

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
)

from .config import MarketingAppConfig
from .simulator import MarketingSimulationResults, evaluate_budget_response_step

FEATURE_NAMES = [
    "Last Month Cust: Competitive",
    "Last Month Cust: Social",
    "Last Month Cust: Trend Explorers",
    "Last Month Cust: Families",
    "Last Month Visits: Competitive",
    "Last Month Visits: Social",
    "Last Month Visits: Trend Explorers",
    "Last Month Visits: Families",
    "Social Media Followers (k)",
    "Follower-to-Cust Conv Rate (~8.57%)",
    "Macro Pickleball Trend Index (I_t)",
    "Active Courts & Venue 2 Headroom",
]


@dataclass
class SegmentMLMetrics:
    """In-sample (80% Train) and Out-of-sample (20% Test) ML regression metrics for 4-segment marketing budget prediction."""

    n_train_obs: int
    n_test_obs: int

    # Overall across all 4 segments
    train_r2: float
    train_rmse_mvnd: float
    train_mae_mvnd: float
    train_mape_pct: float

    test_r2: float
    test_rmse_mvnd: float
    test_mae_mvnd: float
    test_mape_pct: float

    # Per-segment out-of-sample test metrics
    seg_test_r2: dict[str, float]
    seg_test_rmse_mvnd: dict[str, float]
    seg_test_mae_mvnd: dict[str, float]
    seg_test_mape_pct: dict[str, float]

    # Economic Optimality Capture vs Oracle
    oracle_net_contrib_capture_pct: float
    uplift_capture_vs_oracle_pct: float

    # Feature Importances
    feature_names: list[str]
    perm_importance_mean: np.ndarray
    perm_importance_std: np.ndarray


@dataclass
class MarketingMLEvaluationResults:
    """ML predicted 4-segment marketing budgets, resulting visits, customers, and 12-distribution ancillary revenues."""

    ml_metrics: SegmentMLMetrics

    # Full 60-month trajectory arrays under ML Predicted Marketing Policy: shape (N, 60, 4) or (N, 60)
    ml_mkt_budgets_seg: np.ndarray            # (N, 60, 4) M VND
    ml_mkt_budget_total: np.ndarray           # (N, 60) M VND
    ml_followers: np.ndarray                  # (N, 60) followers
    ml_customers_seg: np.ndarray              # (N, 60, 4) customers
    ml_customers_total: np.ndarray            # (N, 60) customers
    ml_visits_seg: np.ndarray                 # (N, 60, 4) visits
    ml_visits_total: np.ndarray               # (N, 60) visits

    # 3 Ancillary Revenue Lines by Segment (12 distributions total) & Totals: shape (N, 60, 4) and (N, 60)
    ml_paddle_rev_seg: np.ndarray             # (N, 60, 4) M VND
    ml_coaching_rev_seg: np.ndarray           # (N, 60, 4) M VND
    ml_fnb_rev_seg: np.ndarray                # (N, 60, 4) M VND
    ml_paddle_rev_total: np.ndarray           # (N, 60) M VND
    ml_coaching_rev_total: np.ndarray         # (N, 60) M VND
    ml_fnb_rev_total: np.ndarray              # (N, 60) M VND
    ml_ancillary_rev_total: np.ndarray        # (N, 60) M VND
    ml_net_contribution: np.ndarray           # (N, 60) M VND

    # Out-of-sample test predictions vs true Oracle for diagnostic scatter plots
    y_test_true: np.ndarray                   # (n_test * 60, 4)
    y_test_pred: np.ndarray                   # (n_test * 60, 4)


def train_and_evaluate_marketing_ml(
    cfg: MarketingAppConfig,
    sim: MarketingSimulationResults,
) -> MarketingMLEvaluationResults:
    """
    Train 4 segment-specific HistGradientBoostingRegressor models on the 80% training split
    (240 trajectories x 60 months = 14,400 observations) to predict this month's optimal
    marketing budget for each of the 4 customer segments from last month's segment customers,
    visits, social followers, conversion rate (~8.57%), trend index, and court capacity.
    """
    n_train = sim.n_train
    N = sim.n_sims
    T = sim.n_months
    S = len(sim.segment_keys)

    X_train = sim.X_features[:n_train].reshape(-1, sim.X_features.shape[-1])
    Y_train = sim.Y_oracle_budgets[:n_train].reshape(-1, S)

    X_test = sim.X_features[n_train:].reshape(-1, sim.X_features.shape[-1])
    Y_test = sim.Y_oracle_budgets[n_train:].reshape(-1, S)

    models: list[HistGradientBoostingRegressor] = []
    Y_train_pred = np.zeros_like(Y_train)
    Y_test_pred = np.zeros_like(Y_test)
    perm_imp_accum = np.zeros((S, X_train.shape[1]), dtype=float)
    perm_std_accum = np.zeros((S, X_train.shape[1]), dtype=float)

    for s_idx in range(S):
        reg = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.065,
            max_depth=6,
            min_samples_leaf=25,
            l2_regularization=1.2,
            random_state=cfg.random_seed + s_idx,
        )
        reg.fit(X_train, Y_train[:, s_idx])
        models.append(reg)

        Y_train_pred[:, s_idx] = np.clip(reg.predict(X_train), 2.0, 24.0)
        Y_test_pred[:, s_idx] = np.clip(reg.predict(X_test), 2.0, 24.0)

        perm = permutation_importance(
            reg,
            X_test,
            Y_test[:, s_idx],
            n_repeats=5,
            random_state=cfg.random_seed + s_idx,
        )
        perm_imp_accum[s_idx] = np.maximum(perm.importances_mean, 0.0)
        perm_std_accum[s_idx] = perm.importances_std

    # Roll out the trained ML models sequentially across all N trajectories x 60 months
    # so that month t's predicted budget drives month t's followers, customers & visits,
    # which then become month (t+1)'s input features!
    ml_mkt_budgets_seg = np.zeros((N, T, S), dtype=float)
    ml_followers = np.zeros((N, T), dtype=float)
    ml_customers_seg = np.zeros((N, T, S), dtype=float)
    ml_visits_seg = np.zeros((N, T, S), dtype=float)
    ml_paddle_rev_seg = np.zeros((N, T, S), dtype=float)
    ml_coaching_rev_seg = np.zeros((N, T, S), dtype=float)
    ml_fnb_rev_seg = np.zeros((N, T, S), dtype=float)
    ml_net_contribution = np.zeros((N, T), dtype=float)

    rng = np.random.default_rng(cfg.random_seed + 999)
    init_followers = cfg.social_funnel.initial_followers_dist.sample(rng, size=N)
    new_follow_rates = cfg.social_funnel.monthly_new_follow_rate_dist.sample(rng, size=(N, T))
    unfollow_rates = cfg.social_funnel.monthly_unfollow_rate_dist.sample(rng, size=(N, T))

    prev_ml_visits = np.tile(np.array([3240.0, 2520.0, 1188.0, 648.0]), (N, 1))
    prev_ml_cust = np.tile(np.array([720.0, 1260.0, 1080.0, 540.0]), (N, 1))
    prev_ml_followers = init_followers.copy()

    for t in range(T):
        m_num = t + 1
        platform_live = 1.0 if m_num >= cfg.membership_launch_month_index else 0.0

        X_step = np.zeros((N, len(FEATURE_NAMES)), dtype=float)
        X_step[:, 0:4] = prev_ml_cust
        X_step[:, 4:8] = prev_ml_visits
        X_step[:, 8] = prev_ml_followers / 1000.0
        X_step[:, 9] = sim.sampled_conv_rate[:, t]
        X_step[:, 10] = sim.sampled_trend_index[:, t]
        X_step[:, 11] = (
            sim.sampled_active_courts[:, t]
            + 10.0 * sim.sampled_venue2_fill_headroom[:, t]
            + 2.0 * platform_live
        )

        for s_idx, reg in enumerate(models):
            ml_mkt_budgets_seg[:, t, s_idx] = np.clip(reg.predict(X_step), 2.0, 24.0)

        (
            m_vis,
            m_cust,
            m_pad,
            m_coach,
            m_fnb,
            m_court,
            m_net_seg,
        ) = evaluate_budget_response_step(
            budgets_mvnd=ml_mkt_budgets_seg[:, t, :],
            base_budgets_mvnd=sim.base_mkt_budgets_seg[:, t, :],
            base_visits_seg=sim.sampled_base_visits_seg[:, t, :],
            elasticity_seg=sim.sampled_elasticity_seg[:, t, :],
            visits_per_cust_seg=sim.sampled_visits_per_cust_seg[:, t, :],
            paddle_kvnd_seg=sim.sampled_paddle_spend_kvnd[:, t, :],
            coaching_kvnd_seg=sim.sampled_coaching_spend_kvnd[:, t, :],
            fnb_kvnd_seg=sim.sampled_fnb_spend_kvnd[:, t, :],
            court_kvnd_seg=sim.sampled_court_spend_kvnd[:, t, :],
            trend_index=sim.sampled_trend_index[:, t],
            active_courts=sim.sampled_active_courts[:, t],
            venue2_headroom=sim.sampled_venue2_fill_headroom[:, t],
            platform_live=platform_live,
            coach_cogs_share=sim.sampled_coach_cogs_share[:, t],
            fnb_cogs_share=sim.sampled_fnb_cogs_share[:, t],
            pay_fee_pct=sim.sampled_pay_fee_pct[:, t],
        )

        mkt_total_ratio = np.sum(ml_mkt_budgets_seg[:, t, :], axis=1) / 25.0
        ml_followers[:, t] = prev_ml_followers * (
            1.0
            - unfollow_rates[:, t]
            + new_follow_rates[:, t]
            * np.power(np.maximum(mkt_total_ratio, 0.5), 0.55)
            * sim.sampled_trend_index[:, t]
        )
        # Explicit Funnel Assumption: Customers = Followers * Follower-to-Customer Conversion Rate (~8.57%)
        funnel_total_customers = ml_followers[:, t] * sim.sampled_conv_rate[:, t]
        seg_cust_shares = m_cust / np.maximum(np.sum(m_cust, axis=1, keepdims=True), 1.0)
        m_cust_aligned = funnel_total_customers[:, None] * seg_cust_shares

        ml_visits_seg[:, t, :] = m_vis
        ml_customers_seg[:, t, :] = m_cust_aligned
        ml_paddle_rev_seg[:, t, :] = m_pad
        ml_coaching_rev_seg[:, t, :] = m_coach
        ml_fnb_rev_seg[:, t, :] = m_fnb
        ml_net_contribution[:, t] = np.sum(m_net_seg, axis=1)

        prev_ml_visits = m_vis
        prev_ml_cust = m_cust_aligned
        prev_ml_followers = ml_followers[:, t]

    # Compute Train and Test Metrics
    train_r2 = float(r2_score(Y_train.ravel(), Y_train_pred.ravel()))
    train_rmse = float(root_mean_squared_error(Y_train.ravel(), Y_train_pred.ravel()))
    train_mae = float(mean_absolute_error(Y_train.ravel(), Y_train_pred.ravel()))
    train_mape = float(mean_absolute_percentage_error(Y_train.ravel(), Y_train_pred.ravel()) * 100.0)

    test_r2 = float(r2_score(Y_test.ravel(), Y_test_pred.ravel()))
    test_rmse = float(root_mean_squared_error(Y_test.ravel(), Y_test_pred.ravel()))
    test_mae = float(mean_absolute_error(Y_test.ravel(), Y_test_pred.ravel()))
    test_mape = float(mean_absolute_percentage_error(Y_test.ravel(), Y_test_pred.ravel()) * 100.0)

    seg_test_r2: dict[str, float] = {}
    seg_test_rmse: dict[str, float] = {}
    seg_test_mae: dict[str, float] = {}
    seg_test_mape: dict[str, float] = {}
    for s_idx, skey in enumerate(sim.segment_keys):
        seg_test_r2[skey] = float(r2_score(Y_test[:, s_idx], Y_test_pred[:, s_idx]))
        seg_test_rmse[skey] = float(root_mean_squared_error(Y_test[:, s_idx], Y_test_pred[:, s_idx]))
        seg_test_mae[skey] = float(mean_absolute_error(Y_test[:, s_idx], Y_test_pred[:, s_idx]))
        seg_test_mape[skey] = float(
            mean_absolute_percentage_error(Y_test[:, s_idx], Y_test_pred[:, s_idx]) * 100.0
        )

    base_test_net = float(np.mean(sim.base_net_contribution[n_train:]))
    oracle_test_net = float(np.mean(sim.oracle_net_contribution[n_train:]))
    ml_test_net = float(np.mean(ml_net_contribution[n_train:]))

    oracle_capture_pct = (ml_test_net / max(oracle_test_net, 1e-6)) * 100.0
    uplift_capture_pct = (
        (ml_test_net - base_test_net) / max(oracle_test_net - base_test_net, 1e-6)
    ) * 100.0

    mean_imp = np.mean(perm_imp_accum, axis=0)
    imp_sum = float(np.sum(mean_imp))
    if imp_sum > 0:
        mean_imp = (mean_imp / imp_sum) * 100.0
    mean_std = np.mean(perm_std_accum, axis=0) * 100.0

    ml_metrics = SegmentMLMetrics(
        n_train_obs=int(X_train.shape[0]),
        n_test_obs=int(X_test.shape[0]),
        train_r2=train_r2,
        train_rmse_mvnd=train_rmse,
        train_mae_mvnd=train_mae,
        train_mape_pct=train_mape,
        test_r2=test_r2,
        test_rmse_mvnd=test_rmse,
        test_mae_mvnd=test_mae,
        test_mape_pct=test_mape,
        seg_test_r2=seg_test_r2,
        seg_test_rmse_mvnd=seg_test_rmse,
        seg_test_mae_mvnd=seg_test_mae,
        seg_test_mape_pct=seg_test_mape,
        oracle_net_contrib_capture_pct=float(np.clip(oracle_capture_pct, 0.0, 100.0)),
        uplift_capture_vs_oracle_pct=float(np.clip(uplift_capture_pct, 0.0, 100.0)),
        feature_names=list(FEATURE_NAMES),
        perm_importance_mean=mean_imp,
        perm_importance_std=mean_std,
    )

    ml_mkt_budget_total = np.sum(ml_mkt_budgets_seg, axis=2)
    ml_customers_total = np.sum(ml_customers_seg, axis=2)
    ml_visits_total = np.sum(ml_visits_seg, axis=2)
    ml_paddle_rev_total = np.sum(ml_paddle_rev_seg, axis=2)
    ml_coaching_rev_total = np.sum(ml_coaching_rev_seg, axis=2)
    ml_fnb_rev_total = np.sum(ml_fnb_rev_seg, axis=2)
    ml_ancillary_rev_total = ml_paddle_rev_total + ml_coaching_rev_total + ml_fnb_rev_total

    return MarketingMLEvaluationResults(
        ml_metrics=ml_metrics,
        ml_mkt_budgets_seg=ml_mkt_budgets_seg,
        ml_mkt_budget_total=ml_mkt_budget_total,
        ml_followers=ml_followers,
        ml_customers_seg=ml_customers_seg,
        ml_customers_total=ml_customers_total,
        ml_visits_seg=ml_visits_seg,
        ml_visits_total=ml_visits_total,
        ml_paddle_rev_seg=ml_paddle_rev_seg,
        ml_coaching_rev_seg=ml_coaching_rev_seg,
        ml_fnb_rev_seg=ml_fnb_rev_seg,
        ml_paddle_rev_total=ml_paddle_rev_total,
        ml_coaching_rev_total=ml_coaching_rev_total,
        ml_fnb_rev_total=ml_fnb_rev_total,
        ml_ancillary_rev_total=ml_ancillary_rev_total,
        ml_net_contribution=ml_net_contribution,
        y_test_true=Y_test,
        y_test_pred=Y_test_pred,
    )
