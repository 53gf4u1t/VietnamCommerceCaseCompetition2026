# Graph Report - CaseComp  (2026-10-31)

## Corpus Check
- Corpus is ~31,206 words - fits in a single context window. You may not need a graph.

## Summary
- 211 nodes - 465 edges - 27 communities (9 shown, 18 thin omitted)
- Extraction: 89% EXTRACTED - 11% INFERRED - 0% AMBIGUOUS - INFERRED: 51 edges (avg confidence: 0.94)
- Token cost: 3,400 input - 1,150 output

## Community Hubs (Navigation)
- 3-Statement Financial Model Engine
- Path C Project Timeline & Gantt Engine
- AdaptivePricing Evaluation & Visualizers
- AdaptivePricing Regime Configuration
- FinancialModel Typed Configuration
- HistGBR ML Demand & Tariff Training
- AdaptivePricing Stage Runner
- AdaptivePricing Report & CSV Exporter
- Cross-Module Reports & Visual Artifacts
- Monte Carlo Court Demand Simulator
- Root Unified Pipeline Orchestrator
- Assumptions & Distributions Chart
- Demand Curves Chart
- Daily Revenue Comparison Chart
- Monthly Calendar Heatmap Chart
- Occupancy & Stress Test Chart
- ML Model Diagnostics Chart
- Sheet 1 Revenue/COGS/OpEx/SGA Render
- Sheet 2 60M 3-Statement Render
- Sheet 3 5Y Annual 3-Statement Render
- CaseComp Workspace Root
- CaseComp README Documentation

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 17 edges
2. `run_adaptive_pricing()` - 16 edges
3. `AppConfig` - 15 edges
4. `load_config()` - 15 edges
5. `run_financial_model()` - 13 edges
6. `FinancialModelConfig` - 13 edges
7. `load_timeline_config()` - 12 edges
8. `EvaluationResults` - 11 edges
9. `train_and_evaluate()` - 11 edges
10. `FinancialModelResults` - 11 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `run_adaptive_pricing()`  [EXTRACTED]
  main.py -> AdaptivePricing/main.py
- `main()` --calls--> `run_financial_model()`  [EXTRACTED]
  main.py -> FinancialModel/main.py
- `main()` --calls--> `run_project_timeline()`  [EXTRACTED]
  main.py -> PickSocialProjectTimeline/main.py
- `Path C 4-Month Build & Migration Estimation Report` --conceptually_related_to--> `5-Year 3-Statement Financial Model Report (FY2027-FY2031)`  [INFERRED]
  PickSocialProjectTimeline/outputs/REPORT.md -> FinancialModel/outputs/REPORT.md
- `train_and_evaluate()` --uses--> `AppConfig`  [INFERRED]
  AdaptivePricing/src/model.py -> AdaptivePricing/src/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **PICK! Unified 3-Module Case Competition Pipeline (Jan 2027 Start)** - adaptivepricing_outputs_report_ml_pricing, picksocialprojecttimeline_outputs_report_timeline, financialmodel_outputs_report_three_statement [EXTRACTED 1.00]

## Communities (27 total, 18 thin omitted)

### Community 0 - "3-Statement Financial Model Engine"
Cohesion: 0.09
Nodes (15): run_financial_model(), FinancialModelConfig, AdaptivePricingIntegration, build_financial_model(), _rec(), CourtRegimeMetrics, FinancialModelResults, LineItemRecord (+7 more)

### Community 1 - "Path C Project Timeline & Gantt Engine"
Cohesion: 0.10
Nodes (17): run_project_timeline(), CitationConfig, load_timeline_config(), MilestoneConfig, ProjectMetaConfig, StageConfig, TaskConfig, TimelineConfig (+9 more)

### Community 2 - "AdaptivePricing Evaluation & Visualizers"
Cohesion: 0.20
Nodes (6): EvaluationResults, plot_assumptions_distributions(), plot_daily_revenue_comparison(), plot_ml_model_diagnostics(), plot_monthly_calendar_heatmap(), plot_occupancy_and_30pct_stress_test()

### Community 3 - "AdaptivePricing Regime Configuration"
Cohesion: 0.19
Nodes (9): AncillaryEconomicsConfig, BaselinePricingConfig, DemandCurveConfig, load_config(), _load_regime(), MLModelConfig, PricingBoundsConfig, RegimeDistributionConfig (+1 more)

### Community 4 - "FinancialModel Typed Configuration"
Cohesion: 0.23
Nodes (11): BalanceSheetConfig, COGSConfig, CompanyConfig, CustomersConfig, CustomerSegmentConfig, load_config(), OperationsConfig, OpExConfig (+3 more)

### Community 5 - "HistGBR ML Demand & Tariff Training"
Cohesion: 0.16
Nodes (3): _build_feature_matrix(), MLTechnicalMetrics, train_and_evaluate()

### Community 6 - "AdaptivePricing Stage Runner"
Cohesion: 0.31
Nodes (3): main(), run_adaptive_pricing(), plot_demand_curve_variations()

### Community 7 - "AdaptivePricing Report & CSV Exporter"
Cohesion: 0.32
Nodes (3): AppConfig, export_tables_and_report(), SimulationResults

### Community 8 - "Cross-Module Reports & Visual Artifacts"
Cohesion: 0.33
Nodes (7): HistGBR Demand Model Metrics, AdaptivePricing ML Optimization Report, Chart 4: Double-Bar Revenue/Costs & Profit Line Comparison, 5-Year 3-Statement Financial Model Report (FY2027-FY2031), Chart 1: Path C Master Production Gantt Chart (Jan-Jun 2027), Chart 2: COCOMO II Effort Allocation & Concurrency Profile, Path C 4-Month Build & Migration Estimation Report

## Knowledge Gaps
- **15 isolated node(s):** `casecomp`, `CaseComp Unified Pipeline`, `HistGBR Demand Model Metrics`, `Assumptions & Demand Distributions Chart`, `Demand & Revenue Curves Chart` (+10 more)
  These have <=1 connection - possible missing edges or undocumented components. (Counts symbols only; 103 node(s) total have <=1 connection when file, concept and rationale nodes are included.)
- **18 thin communities (<3 nodes) omitted from report** - run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `AdaptivePricing Regime Configuration` to `Monte Carlo Court Demand Simulator`, `Path C Project Timeline & Gantt Engine`, `AdaptivePricing Stage Runner`, `AdaptivePricing Report & CSV Exporter`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `AppConfig` (e.g. with `train_and_evaluate()` and `export_tables_and_report()`) actually correct?**
  _`AppConfig` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `casecomp`, `CaseComp Unified Pipeline`, `HistGBR Demand Model Metrics` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `3-Statement Financial Model Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.0851063829787234 - nodes in this community are weakly interconnected._
- **Why does `load_config()` connect `FinancialModel Typed Configuration` to `3-Statement Financial Model Engine`, `Path C Project Timeline & Gantt Engine`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Should `Path C Project Timeline & Gantt Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.10077519379844961 - nodes in this community are weakly interconnected._
- **Why does `run_adaptive_pricing()` connect `AdaptivePricing Stage Runner` to `AdaptivePricing Evaluation & Visualizers`, `AdaptivePricing Regime Configuration`, `HistGBR ML Demand & Tariff Training`, `AdaptivePricing Report & CSV Exporter`, `Monte Carlo Court Demand Simulator`, `Root Unified Pipeline Orchestrator`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._