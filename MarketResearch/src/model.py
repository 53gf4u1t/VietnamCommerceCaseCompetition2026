"""Analytical model for `MarketResearch/` (Competitive Positioning, TAM/SAM/SOM & Customer Demand)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import CompetitorRecord, CustomerPersonaRecord, MarketResearchConfig


@dataclass(frozen=True)
class MarketPositioningAnalysis:
    pick_baseline: CompetitorRecord
    peer_clubs: list[CompetitorRecord]
    cheaper_peak_clubs: list[CompetitorRecord]
    parity_premium_clubs: list[CompetitorRecord]
    avg_peer_peak_kvnd: float
    avg_peer_followers_k: float
    avg_peer_rev_mvnd: float
    pick_follower_multiple_vs_peers: float
    pick_revenue_per_follower_vnd: float
    tam_2026_bvnd: float
    tam_2029_bvnd: float
    sam_2026_bvnd: float
    sam_2029_bvnd: float
    catchment_2026_bvnd: float
    catchment_2029_bvnd: float
    som_2026_bvnd: float
    som_2029_bvnd: float
    som_share_of_sam_2026_pct: float
    som_share_of_sam_2029_pct: float
    som_share_of_catchment_2026_pct: float
    som_share_of_catchment_2029_pct: float
    priority_segments: list[CustomerPersonaRecord]
    total_baseline_customers: int
    total_baseline_visits: int
    total_baseline_revenue_mvnd: float
    total_baseline_marketing_mvnd: float
    priority_two_customer_share_pct: float
    priority_two_visit_share_pct: float


def run_market_research_analysis(cfg: MarketResearchConfig) -> MarketPositioningAnalysis:
    """Compute competitive benchmarking, TAM/SAM/SOM shares, and customer persona priority metrics."""
    pick_base = next(c for c in cfg.competitors if c.id == "pick_baseline")
    peers = [c for c in cfg.competitors if c.id != "pick_baseline"]
    cheaper_peers = [c for c in peers if c.peak_price_kvnd < pick_base.peak_price_kvnd]
    parity_peers = [c for c in peers if c.peak_price_kvnd >= pick_base.peak_price_kvnd]

    avg_peer_peak = float(np.mean([c.peak_price_kvnd for c in peers]))
    avg_peer_foll = float(np.mean([c.social_followers_k for c in peers]))
    avg_peer_rev = float(np.mean([c.annual_revenue_mvnd for c in peers]))

    follower_mult = pick_base.social_followers_k / avg_peer_foll
    rev_per_foll = (pick_base.annual_revenue_mvnd * 1e6) / (pick_base.social_followers_k * 1e3)

    idx_2026 = cfg.sizing.years.index(2026)
    idx_2029 = cfg.sizing.years.index(2029)

    tam_2026 = cfg.sizing.tam_vietnam_bvnd[idx_2026]
    tam_2029 = cfg.sizing.tam_vietnam_bvnd[idx_2029]
    sam_2026 = cfg.sizing.sam_hcmc_bvnd[idx_2026]
    sam_2029 = cfg.sizing.sam_hcmc_bvnd[idx_2029]
    catch_2026 = cfg.sizing.sam_prime_catchment_bvnd[idx_2026]
    catch_2029 = cfg.sizing.sam_prime_catchment_bvnd[idx_2029]
    som_2026 = cfg.sizing.som_pick_bvnd[idx_2026]
    som_2029 = cfg.sizing.som_pick_bvnd[idx_2029]

    som_sam_2026_pct = (som_2026 / sam_2026) * 100.0
    som_sam_2029_pct = (som_2029 / sam_2029) * 100.0
    som_catch_2026_pct = (som_2026 / catch_2026) * 100.0
    som_catch_2029_pct = (som_2029 / catch_2029) * 100.0

    priority_segs = sorted(cfg.personas, key=lambda p: p.priority_rank)
    tot_cust = sum(p.baseline_customers for p in cfg.personas)
    tot_vis = sum(p.baseline_visits_monthly for p in cfg.personas)
    tot_rev = sum(p.baseline_monthly_revenue_mvnd for p in cfg.personas)
    tot_mkt = sum(p.baseline_marketing_budget_mvnd for p in cfg.personas)

    p2_cust_share = sum(p.customer_share_pct for p in cfg.personas if p.id in ("social", "trend_explorers"))
    p2_vis_share = sum(p.visit_share_pct for p in cfg.personas if p.id in ("social", "trend_explorers"))

    return MarketPositioningAnalysis(
        pick_baseline=pick_base,
        peer_clubs=peers,
        cheaper_peak_clubs=cheaper_peers,
        parity_premium_clubs=parity_peers,
        avg_peer_peak_kvnd=avg_peer_peak,
        avg_peer_followers_k=avg_peer_foll,
        avg_peer_rev_mvnd=avg_peer_rev,
        pick_follower_multiple_vs_peers=follower_mult,
        pick_revenue_per_follower_vnd=rev_per_foll,
        tam_2026_bvnd=tam_2026,
        tam_2029_bvnd=tam_2029,
        sam_2026_bvnd=sam_2026,
        sam_2029_bvnd=sam_2029,
        catchment_2026_bvnd=catch_2026,
        catchment_2029_bvnd=catch_2029,
        som_2026_bvnd=som_2026,
        som_2029_bvnd=som_2029,
        som_share_of_sam_2026_pct=som_sam_2026_pct,
        som_share_of_sam_2029_pct=som_sam_2029_pct,
        som_share_of_catchment_2026_pct=som_catch_2026_pct,
        som_share_of_catchment_2029_pct=som_catch_2029_pct,
        priority_segments=priority_segs,
        total_baseline_customers=tot_cust,
        total_baseline_visits=tot_vis,
        total_baseline_revenue_mvnd=tot_rev,
        total_baseline_marketing_mvnd=tot_mkt,
        priority_two_customer_share_pct=p2_cust_share,
        priority_two_visit_share_pct=p2_vis_share,
    )
