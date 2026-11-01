"""Monte Carlo marketing funnel, 4-segment customer response, and 12-distribution ancillary revenue simulator."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import MarketingAppConfig


@dataclass
class MarketingSimulationResults:
    """Complete Monte Carlo simulation tensors across N trajectories x 60 months x 4 customer segments."""

    n_sims: int
    n_train: int
    n_test: int
    n_months: int
    month_labels: tuple[str, ...]
    segment_keys: tuple[str, ...]
    segment_labels: tuple[str, ...]

    # Train/Test Feature Matrix X (n_sims, n_months, n_features=12) & Oracle Target Y (n_sims, n_months, 4)
    X_features: np.ndarray
    Y_oracle_budgets: np.ndarray

    # Baseline (Static 25M/month split: 5.0M, 8.5M, 7.5M, 4.0M) arrays: shape (n_sims, n_months, 4) or (n_sims, n_months)
    base_mkt_budgets_seg: np.ndarray          # (N, 60, 4) M VND
    base_followers: np.ndarray                # (N, 60) followers
    base_conv_rate: np.ndarray                # (N, 60) ~0.0857
    base_customers_total: np.ndarray          # (N, 60) unique customers
    base_customers_seg: np.ndarray            # (N, 60, 4) unique customers by segment
    base_visits_seg: np.ndarray               # (N, 60, 4) player visits by segment
    base_visits_total: np.ndarray             # (N, 60) total visits
    base_paddle_rev_seg: np.ndarray           # (N, 60, 4) M VND
    base_coaching_rev_seg: np.ndarray         # (N, 60, 4) M VND
    base_fnb_rev_seg: np.ndarray              # (N, 60, 4) M VND
    base_ancillary_rev_total: np.ndarray      # (N, 60) M VND
    base_net_contribution: np.ndarray         # (N, 60) M VND

    # Oracle Optimal Marketing arrays (maximizing net segment contribution each month):
    oracle_mkt_budgets_seg: np.ndarray        # (N, 60, 4) M VND
    oracle_followers: np.ndarray              # (N, 60)
    oracle_customers_seg: np.ndarray          # (N, 60, 4)
    oracle_visits_seg: np.ndarray             # (N, 60, 4)
    oracle_paddle_rev_seg: np.ndarray         # (N, 60, 4) M VND
    oracle_coaching_rev_seg: np.ndarray       # (N, 60, 4) M VND
    oracle_fnb_rev_seg: np.ndarray            # (N, 60, 4) M VND
    oracle_net_contribution: np.ndarray       # (N, 60) M VND

    # Sampled stochastic parameters per (N, 60, 4) so the ML evaluator can score any predicted budget vector
    sampled_vnd_per_visit_seg: np.ndarray     # (N, 60, 4) VND/visit
    sampled_elasticity_seg: np.ndarray        # (N, 60, 4) exponent
    sampled_base_visits_seg: np.ndarray       # (N, 60, 4) visits/month at baseline budget
    sampled_visits_per_cust_seg: np.ndarray   # (N, 60, 4) visits/customer
    sampled_paddle_spend_kvnd: np.ndarray     # (N, 60, 4) kVND/visit (4 of 12 distributions)
    sampled_coaching_spend_kvnd: np.ndarray   # (N, 60, 4) kVND/visit (4 of 12 distributions)
    sampled_fnb_spend_kvnd: np.ndarray        # (N, 60, 4) kVND/visit (4 of 12 distributions)
    sampled_court_spend_kvnd: np.ndarray      # (N, 60, 4) kVND/visit
    sampled_conv_rate: np.ndarray             # (N, 60) ~8.57% follower-to-customer conversion
    sampled_trend_index: np.ndarray           # (N, 60) cumulative pickleball interest index I_t
    sampled_active_courts: np.ndarray         # (N, 60) 6 or 20 courts
    sampled_venue2_fill_headroom: np.ndarray  # (N, 60) 0..1 headroom for filling new 14 courts
    sampled_coach_cogs_share: np.ndarray      # (N, 60) ~0.5217
    sampled_fnb_cogs_share: np.ndarray        # (N, 60) ~0.4000
    sampled_pay_fee_pct: np.ndarray           # (N, 60) ~0.0150


def _build_month_labels(start_label: str, n_months: int) -> tuple[str, ...]:
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    parts = start_label.split("-")
    m_idx = month_names.index(parts[0])
    yr = int(parts[1])
    labels: list[str] = []
    for i in range(n_months):
        cur_m = (m_idx + i) % 12
        cur_y = yr + (m_idx + i) // 12
        labels.append(f"{month_names[cur_m]}-{cur_y:02d}")
    return tuple(labels)


def _sample_nt_shared(
    dist,
    rng: np.random.Generator,
    n_sims: int,
    n_months: int,
    shared_weight: float = 0.58,
) -> np.ndarray:
    """
    Sample an (n_sims, n_months) tensor from `dist` combining a shared calendar-month
    realized environment draw (`shared_weight`) and an independent trajectory draw
    (`sqrt(1 - shared_weight^2)`), preserving 100% of `dist.mean`, `dist.std`, and `[min, max]`.
    """
    if dist.std <= 0.0:
        return np.full((n_sims, n_months), dist.mean, dtype=float)
    z_month = rng.standard_normal(size=(1, n_months))
    z_sims = rng.standard_normal(size=(n_sims, n_months))
    idio_weight = float(np.sqrt(max(0.0, 1.0 - shared_weight ** 2)))
    raw = dist.mean + dist.std * (shared_weight * z_month + idio_weight * z_sims)
    return np.clip(raw, dist.min, dist.max)


def evaluate_budget_response_step(
    budgets_mvnd: np.ndarray,
    base_budgets_mvnd: np.ndarray,
    base_visits_seg: np.ndarray,
    elasticity_seg: np.ndarray,
    visits_per_cust_seg: np.ndarray,
    paddle_kvnd_seg: np.ndarray,
    coaching_kvnd_seg: np.ndarray,
    fnb_kvnd_seg: np.ndarray,
    court_kvnd_seg: np.ndarray,
    trend_index: float | np.ndarray,
    active_courts: float | np.ndarray,
    venue2_headroom: float | np.ndarray,
    platform_live: float | np.ndarray,
    coach_cogs_share: float | np.ndarray,
    fnb_cogs_share: float | np.ndarray,
    pay_fee_pct: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate segment visits, customers, 3 ancillary revenue lines (Paddle/Balls, Coaching, F&B),
    and net contribution for a given 4-segment marketing budget vector `budgets_mvnd`.
    Works on either a single simulation step `(4,)` or vectorized across `(N, 4)`.
    """
    # 1. Capacity multiplier: when Venue 2 (14 new courts -> 20 total courts) opens in Month 11,
    #    addressable court-hour capacity expands from 6 courts to 20 courts (up to 3.33x),
    #    modulated by the progressive fill-up ramp and platform matchmaking efficiency.
    is_expanded = np.where(np.asarray(active_courts) > 6.0, 1.0, 0.0)
    # Effective addressable capacity multiplier relative to the 6-court baseline (1.0 at 6 courts,
    # scaling smoothly up to ~2.35x across both clubs as Venue 2 fills up):
    fill_progress = np.where(is_expanded > 0.5, 1.0 - np.asarray(venue2_headroom), 0.0)
    capacity_regime_boost = 1.0 + is_expanded * (0.42 + 0.90 * fill_progress) + 0.08 * np.asarray(platform_live)

    # 2. Segment marketing response curve (concave power-law elasticity around baseline budget B_{0,s}):
    #    At B_{s,t} == B_{0,s} and 6 courts (trend=1.0), response ratio = 1.0 -> exact baseline visits!
    budget_ratio = np.maximum(budgets_mvnd, 0.50) / np.maximum(base_budgets_mvnd, 0.50)
    # Concave diminishing returns response: 1 + (1/elasticity_s) * ((B/B0)^elasticity_s - 1) * 0.48
    mkt_response_mult = 1.0 + 0.52 * (np.power(budget_ratio, elasticity_seg) - 1.0)

    # Physical court saturation soft-cap so visits cannot exceed physical court-hour capacity
    max_cap_mult = np.where(is_expanded > 0.5, 2.85, 1.24)
    if np.ndim(budgets_mvnd) == 2:
        trend_col = np.asarray(trend_index)[:, None]
        cap_boost_col = np.asarray(capacity_regime_boost)[:, None]
        max_cap_col = np.asarray(max_cap_mult)[:, None]
    else:
        trend_col = float(trend_index)
        cap_boost_col = float(capacity_regime_boost)
        max_cap_col = float(max_cap_mult)

    effective_visit_mult = np.minimum(max_cap_col, mkt_response_mult * trend_col * cap_boost_col)
    visits_seg = base_visits_seg * effective_visit_mult
    customers_seg = visits_seg / np.maximum(visits_per_cust_seg, 0.5)

    # 3. Ancillary revenues across the 12 distributions (in Million VND = visits * kVND / 1,000):
    paddle_rev_seg = (visits_seg * paddle_kvnd_seg) / 1000.0
    coaching_rev_seg = (visits_seg * coaching_kvnd_seg) / 1000.0
    fnb_rev_seg = (visits_seg * fnb_kvnd_seg) / 1000.0
    court_rev_seg = (visits_seg * court_kvnd_seg) / 1000.0

    # 4. Net contribution margin per segment after variable COGS (Coach fees ~52.17%, F&B supplies ~40%,
    #    Payment fees ~1.5%) and after subtracting the segment marketing spend `budgets_mvnd`:
    if np.ndim(budgets_mvnd) == 2:
        c_share = np.asarray(coach_cogs_share)[:, None]
        f_share = np.asarray(fnb_cogs_share)[:, None]
        p_fee = np.asarray(pay_fee_pct)[:, None]
    else:
        c_share = float(coach_cogs_share)
        f_share = float(fnb_cogs_share)
        p_fee = float(pay_fee_pct)

    gross_rev_seg = court_rev_seg + paddle_rev_seg + coaching_rev_seg + fnb_rev_seg
    var_cogs_seg = (
        coaching_rev_seg * c_share
        + fnb_rev_seg * f_share
        + gross_rev_seg * p_fee
        + paddle_rev_seg * 0.12
    )
    # Regularity / LTV retention bonus (Competitive & Social have higher repeat & subscription LTV than Trend Explorers)
    ltv_weights = np.array([1.22, 1.18, 0.78, 1.12], dtype=float)
    net_contrib_seg = (gross_rev_seg - var_cogs_seg) * ltv_weights - budgets_mvnd
    # Quadratic budget Governance penalty for extreme single-month budget jumps
    reg_penalty = 0.032 * np.square(budgets_mvnd - base_budgets_mvnd * cap_boost_col)
    net_obj_seg = net_contrib_seg - reg_penalty

    return (
        visits_seg,
        customers_seg,
        paddle_rev_seg,
        coaching_rev_seg,
        fnb_rev_seg,
        court_rev_seg,
        net_obj_seg,
    )


def run_marketing_simulation(cfg: MarketingAppConfig) -> MarketingSimulationResults:
    """
    Simulate N=300 60-month trajectories of the social follower funnel (~8.57% conversion),
    4 customer segments, 12 ancillary spending distributions, and compute the Oracle optimal
    monthly segment marketing budgets for training & evaluating our ML Marketing Budget Predictor.
    """
    rng = np.random.default_rng(cfg.random_seed)
    N = cfg.n_simulations
    n_train = int(round(N * cfg.train_fraction))
    n_test = N - n_train
    T = cfg.projection_months
    S = len(cfg.segments)  # 4 segments

    month_labels = _build_month_labels(cfg.model_start_month, T)
    segment_keys = tuple(s.key for s in cfg.segments)
    segment_labels = tuple(s.label for s in cfg.segments)

    # Sample stochastic distributions across (N, T, S)
    sampled_base_visits_seg = np.zeros((N, T, S), dtype=float)
    sampled_base_budgets_seg = np.zeros((N, T, S), dtype=float)
    sampled_vnd_per_visit_seg = np.zeros((N, T, S), dtype=float)
    sampled_elasticity_seg = np.zeros((N, T, S), dtype=float)
    sampled_visits_per_cust_seg = np.zeros((N, T, S), dtype=float)
    sampled_paddle_spend_kvnd = np.zeros((N, T, S), dtype=float)
    sampled_coaching_spend_kvnd = np.zeros((N, T, S), dtype=float)
    sampled_fnb_spend_kvnd = np.zeros((N, T, S), dtype=float)
    sampled_court_spend_kvnd = np.zeros((N, T, S), dtype=float)

    for s_idx, seg in enumerate(cfg.segments):
        sampled_base_visits_seg[:, :, s_idx] = _sample_nt_shared(seg.monthly_visits_dist, rng, N, T, shared_weight=0.58)
        sampled_base_budgets_seg[:, :, s_idx] = _sample_nt_shared(seg.baseline_marketing_budget_mvnd_dist, rng, N, T, shared_weight=0.58)
        sampled_vnd_per_visit_seg[:, :, s_idx] = _sample_nt_shared(seg.marketing_vnd_per_visit_dist, rng, N, T, shared_weight=0.58)
        sampled_elasticity_seg[:, :, s_idx] = _sample_nt_shared(seg.marketing_elasticity_dist, rng, N, T, shared_weight=0.50)
        sampled_visits_per_cust_seg[:, :, s_idx] = _sample_nt_shared(seg.visits_per_month_dist, rng, N, T, shared_weight=0.50)
        # The 12 Segment-by-Ancillary Spending Distributions (3 ancillary lines x 4 segments):
        sampled_paddle_spend_kvnd[:, :, s_idx] = _sample_nt_shared(seg.paddle_balls_spend_per_visit_kvnd_dist, rng, N, T, shared_weight=0.58)
        sampled_coaching_spend_kvnd[:, :, s_idx] = _sample_nt_shared(seg.coaching_spend_per_visit_kvnd_dist, rng, N, T, shared_weight=0.58)
        sampled_fnb_spend_kvnd[:, :, s_idx] = _sample_nt_shared(seg.fnb_spend_per_visit_kvnd_dist, rng, N, T, shared_weight=0.58)
        sampled_court_spend_kvnd[:, :, s_idx] = _sample_nt_shared(seg.court_hire_spend_per_visit_kvnd_dist, rng, N, T, shared_weight=0.58)

    # Sample social funnel (~8.57% follower-to-customer conversion), trend, and COGS parameters
    init_followers = cfg.social_funnel.initial_followers_dist.sample(rng, size=N)
    sampled_conv_rate = _sample_nt_shared(cfg.social_funnel.follower_to_customer_conversion_dist, rng, N, T, shared_weight=0.55)
    new_follow_rates = _sample_nt_shared(cfg.social_funnel.monthly_new_follow_rate_dist, rng, N, T, shared_weight=0.55)
    unfollow_rates = _sample_nt_shared(cfg.social_funnel.monthly_unfollow_rate_dist, rng, N, T, shared_weight=0.55)

    trend_changes = _sample_nt_shared(cfg.trend_monthly_change_dist, rng, N, T, shared_weight=0.55)
    sampled_trend_index = np.ones((N, T), dtype=float)
    for t in range(1, T):
        sampled_trend_index[:, t] = np.clip(
            sampled_trend_index[:, t - 1] * (1.0 + trend_changes[:, t]),
            cfg.trend_floor,
            cfg.trend_ceiling,
        )

    v2_fill_durations = cfg.venue2_fill_up_months_dist.sample(rng, size=N)
    sampled_active_courts = np.zeros((N, T), dtype=float)
    sampled_venue2_fill_headroom = np.zeros((N, T), dtype=float)
    for t in range(T):
        m_num = t + 1
        if m_num < cfg.venue2_open_month_index:
            sampled_active_courts[:, t] = 6.0
            sampled_venue2_fill_headroom[:, t] = 0.0
        else:
            sampled_active_courts[:, t] = 20.0
            m_since_open = float(m_num - cfg.venue2_open_month_index + 1)
            x_prog = m_since_open / np.maximum(v2_fill_durations, 1.0)
            fill_f = np.clip(0.22 + 0.78 * (1.0 - np.exp(-2.6 * x_prog)) / (1.0 - np.exp(-2.6)), 0.22, 1.0)
            sampled_venue2_fill_headroom[:, t] = 1.0 - fill_f

    sampled_coach_cogs_share = cfg.coach_fees_share_dist.sample(rng, size=(N, T))
    sampled_fnb_cogs_share = cfg.fnb_supplies_share_dist.sample(rng, size=(N, T))
    sampled_pay_fee_pct = cfg.payment_fee_pct_dist.sample(rng, size=(N, T))

    # Allocate trajectory arrays for Baseline vs Oracle Optimal
    base_mkt_budgets_seg = sampled_base_budgets_seg.copy()
    base_followers = np.zeros((N, T), dtype=float)
    base_customers_seg = np.zeros((N, T, S), dtype=float)
    base_visits_seg = np.zeros((N, T, S), dtype=float)
    base_paddle_rev_seg = np.zeros((N, T, S), dtype=float)
    base_coaching_rev_seg = np.zeros((N, T, S), dtype=float)
    base_fnb_rev_seg = np.zeros((N, T, S), dtype=float)
    base_net_contribution = np.zeros((N, T), dtype=float)

    oracle_mkt_budgets_seg = np.zeros((N, T, S), dtype=float)
    oracle_followers = np.zeros((N, T), dtype=float)
    oracle_customers_seg = np.zeros((N, T, S), dtype=float)
    oracle_visits_seg = np.zeros((N, T, S), dtype=float)
    oracle_paddle_rev_seg = np.zeros((N, T, S), dtype=float)
    oracle_coaching_rev_seg = np.zeros((N, T, S), dtype=float)
    oracle_fnb_rev_seg = np.zeros((N, T, S), dtype=float)
    oracle_net_contribution = np.zeros((N, T), dtype=float)

    # Feature matrix X: 12 features per month t based on last month (t-1) segment customers,
    # visits, followers, conversion rate, trend index, active courts, and Venue 2 headroom
    n_features = 12
    X_features = np.zeros((N, T, n_features), dtype=float)

    # Candidate budget grid per segment (from 2.0M to 24.0M VND in 0.25M steps = 89 points)
    budget_grid = np.linspace(2.0, 24.0, 89)

    prev_oracle_visits = np.tile(np.array([3240.0, 2520.0, 1188.0, 648.0]), (N, 1))
    prev_oracle_cust = np.tile(np.array([720.0, 1260.0, 1080.0, 540.0]), (N, 1))
    prev_oracle_followers = init_followers.copy()
    prev_base_followers = init_followers.copy()

    for t in range(T):
        m_num = t + 1
        platform_live = 1.0 if m_num >= cfg.membership_launch_month_index else 0.0

        # Build Feature Vector X[:, t, :] from last month's state:
        # [0..3]: Last month's customers in each of the 4 segments (Competitive, Social, Trend Explorers, Families)
        # [4..7]: Last month's visits in each of the 4 segments
        # [8]   : Last month's social media followers (in thousands)
        # [9]   : Follower-to-customer conversion rate (~8.57%)
        # [10]  : Macro pickleball trend index I_t
        # [11]  : Active court capacity (6 vs 20) + Venue 2 headroom composite
        X_features[:, t, 0:4] = prev_oracle_cust
        X_features[:, t, 4:8] = prev_oracle_visits
        X_features[:, t, 8] = prev_oracle_followers / 1000.0
        X_features[:, t, 9] = sampled_conv_rate[:, t]
        X_features[:, t, 10] = sampled_trend_index[:, t]
        X_features[:, t, 11] = sampled_active_courts[:, t] + 10.0 * sampled_venue2_fill_headroom[:, t] + 2.0 * platform_live

        # 1. Evaluate Baseline (static ~25M VND/mo split: 5.0M, 8.5M, 7.5M, 4.0M)
        (
            b_vis,
            b_cust,
            b_pad,
            b_coach,
            b_fnb,
            b_court,
            b_net_seg,
        ) = evaluate_budget_response_step(
            budgets_mvnd=base_mkt_budgets_seg[:, t, :],
            base_budgets_mvnd=sampled_base_budgets_seg[:, t, :],
            base_visits_seg=sampled_base_visits_seg[:, t, :],
            elasticity_seg=sampled_elasticity_seg[:, t, :],
            visits_per_cust_seg=sampled_visits_per_cust_seg[:, t, :],
            paddle_kvnd_seg=sampled_paddle_spend_kvnd[:, t, :],
            coaching_kvnd_seg=sampled_coaching_spend_kvnd[:, t, :],
            fnb_kvnd_seg=sampled_fnb_spend_kvnd[:, t, :],
            court_kvnd_seg=sampled_court_spend_kvnd[:, t, :],
            trend_index=sampled_trend_index[:, t],
            active_courts=sampled_active_courts[:, t],
            venue2_headroom=sampled_venue2_fill_headroom[:, t],
            platform_live=platform_live,
            coach_cogs_share=sampled_coach_cogs_share[:, t],
            fnb_cogs_share=sampled_fnb_cogs_share[:, t],
            pay_fee_pct=sampled_pay_fee_pct[:, t],
        )
        base_visits_seg[:, t, :] = b_vis
        # Customer count grounded in our explicit assumption: Followers * Conversion Rate (~8.57%)
        base_followers[:, t] = prev_base_followers * (
            1.0 - unfollow_rates[:, t] + new_follow_rates[:, t] * sampled_trend_index[:, t]
        )
        prev_base_followers = base_followers[:, t]
        base_customers_seg[:, t, :] = b_cust
        base_paddle_rev_seg[:, t, :] = b_pad
        base_coaching_rev_seg[:, t, :] = b_coach
        base_fnb_rev_seg[:, t, :] = b_fnb
        base_net_contribution[:, t] = np.sum(b_net_seg, axis=1)

        # 2. Find Oracle Optimal 4-Segment Marketing Budget for Month t across the grid
        best_budgets_t = np.zeros((N, S), dtype=float)
        for s_idx in range(S):
            # Evaluate all 89 candidate budgets for segment s_idx simultaneously
            # Shape: (N, 89)
            cand_b = budget_grid[None, :]
            b0 = sampled_base_budgets_seg[:, t, s_idx : s_idx + 1]
            v0 = sampled_base_visits_seg[:, t, s_idx : s_idx + 1]
            el = sampled_elasticity_seg[:, t, s_idx : s_idx + 1]
            pad_k = sampled_paddle_spend_kvnd[:, t, s_idx : s_idx + 1]
            coa_k = sampled_coaching_spend_kvnd[:, t, s_idx : s_idx + 1]
            fnb_k = sampled_fnb_spend_kvnd[:, t, s_idx : s_idx + 1]
            crt_k = sampled_court_spend_kvnd[:, t, s_idx : s_idx + 1]

            is_exp = np.where(sampled_active_courts[:, t : t + 1] > 6.0, 1.0, 0.0)
            fill_p = np.where(is_exp > 0.5, 1.0 - sampled_venue2_fill_headroom[:, t : t + 1], 0.0)
            cap_boost = 1.0 + is_exp * (0.42 + 0.90 * fill_p) + 0.08 * platform_live
            # During Venue 2 pre-launch & early ramp (M10-M18), marketing has higher strategic fill-up value
            ramp_urgency = np.where((m_num >= 10) & (m_num <= 18), 1.22, 1.0)
            if m_num >= 5 and m_num <= 9:
                ramp_urgency = 1.10  # Platform launch subscriber acquisition window

            b_ratio = np.maximum(cand_b, 0.50) / np.maximum(b0, 0.50)
            mkt_mult = 1.0 + 0.52 * (np.power(b_ratio, el) - 1.0)
            max_c = np.where(is_exp > 0.5, 2.85, 1.24)
            eff_mult = np.minimum(max_c, mkt_mult * sampled_trend_index[:, t : t + 1] * cap_boost)
            vis_cand = v0 * eff_mult

            pad_r = (vis_cand * pad_k) / 1000.0
            coa_r = (vis_cand * coa_k) / 1000.0
            fnb_r = (vis_cand * fnb_k) / 1000.0
            crt_r = (vis_cand * crt_k) / 1000.0
            gross_r = crt_r + pad_r + coa_r + fnb_r
            var_c = (
                coa_r * sampled_coach_cogs_share[:, t : t + 1]
                + fnb_r * sampled_fnb_cogs_share[:, t : t + 1]
                + gross_r * sampled_pay_fee_pct[:, t : t + 1]
                + pad_r * 0.12
            )
            ltv_w = [1.22, 1.18, 0.78, 1.12][s_idx] * ramp_urgency
            reg_pen = 0.032 * np.square(cand_b - b0 * cap_boost)
            obj_cand = (gross_r - var_c) * ltv_w - cand_b - reg_pen

            best_idx = np.argmax(obj_cand, axis=1)
            best_budgets_t[:, s_idx] = budget_grid[best_idx]

        oracle_mkt_budgets_seg[:, t, :] = best_budgets_t
        (
            o_vis,
            o_cust,
            o_pad,
            o_coach,
            o_fnb,
            o_court,
            o_net_seg,
        ) = evaluate_budget_response_step(
            budgets_mvnd=oracle_mkt_budgets_seg[:, t, :],
            base_budgets_mvnd=sampled_base_budgets_seg[:, t, :],
            base_visits_seg=sampled_base_visits_seg[:, t, :],
            elasticity_seg=sampled_elasticity_seg[:, t, :],
            visits_per_cust_seg=sampled_visits_per_cust_seg[:, t, :],
            paddle_kvnd_seg=sampled_paddle_spend_kvnd[:, t, :],
            coaching_kvnd_seg=sampled_coaching_spend_kvnd[:, t, :],
            fnb_kvnd_seg=sampled_fnb_spend_kvnd[:, t, :],
            court_kvnd_seg=sampled_court_spend_kvnd[:, t, :],
            trend_index=sampled_trend_index[:, t],
            active_courts=sampled_active_courts[:, t],
            venue2_headroom=sampled_venue2_fill_headroom[:, t],
            platform_live=platform_live,
            coach_cogs_share=sampled_coach_cogs_share[:, t],
            fnb_cogs_share=sampled_fnb_cogs_share[:, t],
            pay_fee_pct=sampled_pay_fee_pct[:, t],
        )

        # Social followers grow with total marketing spend relative to 25M baseline, and
        # active unique customers equal Followers * Conversion Rate (~8.57% distribution)
        cal_m = (t % 12) + 1
        is_tourney_m = cfg.tournament_enabled and (m_num >= cfg.tournament_start_month_index) and (cal_m in cfg.tournament_calendar_months)
        fol_boost = cfg.tournament_follower_boost_mult if is_tourney_m else 1.0

        mkt_total_ratio = np.sum(oracle_mkt_budgets_seg[:, t, :], axis=1) / 25.0
        oracle_followers[:, t] = prev_oracle_followers * (
            1.0
            - unfollow_rates[:, t]
            + new_follow_rates[:, t] * fol_boost * np.power(np.maximum(mkt_total_ratio, 0.5), 0.55) * sampled_trend_index[:, t]
        )
        # Align segment customer count with our explicit assumption: Customers = Followers * ConvRate (~8.57%)
        funnel_total_customers = oracle_followers[:, t] * sampled_conv_rate[:, t]
        seg_cust_shares = o_cust / np.maximum(np.sum(o_cust, axis=1, keepdims=True), 1.0)
        o_cust_aligned = funnel_total_customers[:, None] * seg_cust_shares

        if is_tourney_m:
            # Tournament-weekend incremental ancillary spend across all participants (members + non-members)
            t_players = (
                cfg.tournament_participants_6_courts_dist.sample(rng, size=N)
                if m_num < cfg.venue2_open_month_index
                else cfg.tournament_participants_20_courts_dist.sample(rng, size=N) * (1.0 - 0.35 * sampled_venue2_fill_headroom[:, t])
            )
            t_seg_weights = np.array([0.45, 0.35, 0.12, 0.08], dtype=float)[None, :]
            t_players_seg = t_players[:, None] * t_seg_weights
            o_vis = o_vis + t_players_seg
            o_pad = o_pad + (t_players_seg * cfg.tournament_paddle_per_participant_kvnd_dist.sample(rng, size=(N, 1))) / 1000.0
            o_coach = o_coach + (t_players_seg * cfg.tournament_coaching_per_participant_kvnd_dist.sample(rng, size=(N, 1))) / 1000.0
            o_fnb = o_fnb + (t_players_seg * cfg.tournament_fnb_per_participant_kvnd_dist.sample(rng, size=(N, 1))) / 1000.0

        oracle_visits_seg[:, t, :] = o_vis
        oracle_customers_seg[:, t, :] = o_cust_aligned
        oracle_paddle_rev_seg[:, t, :] = o_pad
        oracle_coaching_rev_seg[:, t, :] = o_coach
        oracle_fnb_rev_seg[:, t, :] = o_fnb
        oracle_net_contribution[:, t] = np.sum(o_net_seg, axis=1)

        prev_oracle_visits = o_vis
        prev_oracle_cust = o_cust_aligned
        prev_oracle_followers = oracle_followers[:, t]

    base_visits_total = np.sum(base_visits_seg, axis=2)
    base_customers_total = base_followers * sampled_conv_rate
    base_ancillary_rev_total = np.sum(
        base_paddle_rev_seg + base_coaching_rev_seg + base_fnb_rev_seg, axis=2
    )

    return MarketingSimulationResults(
        n_sims=N,
        n_train=n_train,
        n_test=n_test,
        n_months=T,
        month_labels=month_labels,
        segment_keys=segment_keys,
        segment_labels=segment_labels,
        X_features=X_features,
        Y_oracle_budgets=oracle_mkt_budgets_seg,
        base_mkt_budgets_seg=base_mkt_budgets_seg,
        base_followers=base_followers,
        base_conv_rate=sampled_conv_rate,
        base_customers_total=base_customers_total,
        base_customers_seg=base_customers_seg,
        base_visits_seg=base_visits_seg,
        base_visits_total=base_visits_total,
        base_paddle_rev_seg=base_paddle_rev_seg,
        base_coaching_rev_seg=base_coaching_rev_seg,
        base_fnb_rev_seg=base_fnb_rev_seg,
        base_ancillary_rev_total=base_ancillary_rev_total,
        base_net_contribution=base_net_contribution,
        oracle_mkt_budgets_seg=oracle_mkt_budgets_seg,
        oracle_followers=oracle_followers,
        oracle_customers_seg=oracle_customers_seg,
        oracle_visits_seg=oracle_visits_seg,
        oracle_paddle_rev_seg=oracle_paddle_rev_seg,
        oracle_coaching_rev_seg=oracle_coaching_rev_seg,
        oracle_fnb_rev_seg=oracle_fnb_rev_seg,
        oracle_net_contribution=oracle_net_contribution,
        sampled_vnd_per_visit_seg=sampled_vnd_per_visit_seg,
        sampled_elasticity_seg=sampled_elasticity_seg,
        sampled_base_visits_seg=sampled_base_visits_seg,
        sampled_visits_per_cust_seg=sampled_visits_per_cust_seg,
        sampled_paddle_spend_kvnd=sampled_paddle_spend_kvnd,
        sampled_coaching_spend_kvnd=sampled_coaching_spend_kvnd,
        sampled_fnb_spend_kvnd=sampled_fnb_spend_kvnd,
        sampled_court_spend_kvnd=sampled_court_spend_kvnd,
        sampled_conv_rate=sampled_conv_rate,
        sampled_trend_index=sampled_trend_index,
        sampled_active_courts=sampled_active_courts,
        sampled_venue2_fill_headroom=sampled_venue2_fill_headroom,
        sampled_coach_cogs_share=sampled_coach_cogs_share,
        sampled_fnb_cogs_share=sampled_fnb_cogs_share,
        sampled_pay_fee_pct=sampled_pay_fee_pct,
    )
