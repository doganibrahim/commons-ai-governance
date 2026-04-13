# Commons Simulation

Commons Simulation is an agent-based model for studying shared-resource governance (commons), social behavior under scarcity, and the behavioral impact of governance technologies.

The repository combines three practical workflows:
- interactive simulation with Mesa + Solara visualization
- large-scale batch experimentation across scenario matrices
- statistical post-analysis with ANOVA and Tukey HSD summaries

## Table of Contents

- [Project Goals](#project-goals)
- [What This Repository Implements](#what-this-repository-implements)
- [Conceptual Background](#conceptual-background)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Interactive Simulation](#interactive-simulation)
- [Batch Experiments](#batch-experiments)
- [Statistical Analysis and Reporting](#statistical-analysis-and-reporting)
- [Scenario Design and Key Parameters](#scenario-design-and-key-parameters)
- [Output Files](#output-files)
- [Metrics Dictionary](#metrics-dictionary)
- [Governance Modes](#governance-modes)
- [Model Execution Loop](#model-execution-loop)
- [Reproducibility Notes](#reproducibility-notes)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)
- [Roadmap Ideas](#roadmap-ideas)
- [License](#license)

## Project Goals

This project is designed to explore questions such as:

- How do different governance modes (AI, blockchain, integrated modes) affect cooperation?
- Under which conditions does free-rider behavior increase or decrease?
- How do trust, autonomy, satisfaction, and fairness evolve over time?
- How do inequality (Gini), sustainability, and conflict dynamics change across scenarios?

## What This Repository Implements

- **Agent-based simulation:** interaction between `PersonAgent` and `ResourceAgent`.
- **Heterogeneous agent types:** `ideal`, `standard`, `toxic`.
- **Behavioral decision modeling:** trust, autonomy, satisfaction, scarcity perception, and dynamic cooperation probabilities.
- **Multiple governance modes:** baseline, AI-assisted, blockchain, and integrated variants.
- **Anomaly detection:** periodic free-rider detection via Isolation Forest.
- **Sanction logic:** temporary request blocking for anomalies in specific modes.
- **Large experimental matrix:** 104 scenarios x 25 repeats x 300 steps.
- **Academic-style analysis output:** ANOVA table, eta-squared effect size, and Tukey pairwise comparisons.

## Conceptual Background

The model references and combines ideas from:

- Commons governance (inspired by Ostrom-style resource sharing principles)
- Self-Determination concepts (autonomy and perceived decision cost)
- Equity Theory (fairness from neighbor-level wait-time comparisons)
- Feedback learning (positive/negative experiences updating trust and satisfaction)

This allows both micro-level behavioral dynamics and macro-level system outcomes to be observed simultaneously.

## Repository Structure

```text
commons-simulation/
├─ agents/
│  ├─ person.py
│  └─ resource.py
├─ model/
│  └─ model.py
├─ modules/
│  └─ psychology.py
├─ run.py
├─ batch_run.py
├─ analysis_report.py
└─ anova_analysis.ipynb
```

## Setup

### Requirements

- Python 3.10+ (recommended)
- `pip`

### 1) Clone the repository

```bash
git clone <REPO_URL>
cd commons-simulation
```

### 2) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

If no `requirements.txt` is provided, install the minimum packages explicitly used by the code:

```bash
pip install mesa solara pandas scikit-learn statsmodels
```

## Quick Start

### Interactive simulation

Because `run.py` defines a Solara page object, the most reliable launch command is:

```bash
solara run run.py
```

In some environments, `python run.py` may also work depending on Mesa/Solara versions and local setup.

### Batch experiments

```bash
python batch_run.py
```

### Statistical report generation

```bash
python analysis_report.py
```

## Interactive Simulation

`run.py` builds a `SolaraViz` page around `CommonsModel` and includes multiple live plot panels.

Agent colors in the space view:
- **Resource agents (`ResourceAgent`)**
  - green: free
  - red: occupied
- **Person agents (`PersonAgent`)**
  - blue: currently using a resource
  - orange: currently defecting
  - gray: neutral/default state

Displayed model metrics include:
- `mean_trust`, `mean_autonomy`, `mean_satisfaction`
- `gini_coefficient`, `resource_utilization`, `sustainability_index`
- `cooperation_rate`, `free_rider_ratio`
- `iforest_anomaly_ratio`, `sanction_rate`, `conflict_rate`

## Batch Experiments

`batch_run.py` is configured for:

- **8 system configurations** (config names)
- **13 context profiles**
- **104 scenarios total**
- **25 repeats per scenario**
- **300 steps per repeat**

Total runs:

`104 x 25 = 2600`

The script:
- creates and validates the scenario matrix
- runs all configured simulations
- appends model and agent-level metrics into CSV files under `output/`

### Quick smoke test

For a short run, use:

```python
run_experiments(max_runs=2)
```

## Statistical Analysis and Reporting

`analysis_report.py`:

1. Reads `output/model_metrics_104x25.csv` and `output/scenario_matrix_104.csv`.
2. Extracts final-tick rows per `scenario_id + repeat_id`.
3. Fits ANOVA models:
   - `metric ~ C(system_type) + C(cohesion) + N_people + N_resources`
4. Computes eta-squared effect sizes.
5. Runs Tukey HSD pairwise comparisons by `system_type`.
6. Writes a markdown report to `output/academic_report_summary.md`.

Current core metrics analyzed:
- `gini_coefficient`
- `mean_trust`
- `resource_utilization`
- `free_rider_ratio`

## Scenario Design and Key Parameters

Context profiles combine:
- `N_people`
- `N_resources`
- `base_trust_shift`
- `cohesion`
- `agent_type_distribution` (`ideal`, `standard`, `toxic`)

Model-level parameters include:
- `system_type`
- `procedural_bonus_modifier`
- `random_seed`
- `detect_every_n_steps` (default: 10)
- `anomaly_contamination` (default: 0.15)

Important implementation note:
- `base_trust_shift` is currently exported as scenario metadata and carried into output tables, but it is not yet applied directly in `CommonsModel` initialization logic.

## Output Files

After running the pipelines, expected artifacts under `output/`:

- `scenario_matrix_104.csv`  
  Scenario definitions and distribution columns.
- `model_metrics_104x25.csv`  
  Step-level model metrics.
- `agent_metrics_104x25.csv`  
  Agent-level metrics across runs.
- `academic_report_summary.md`  
  ANOVA/Tukey summary report.

## Metrics Dictionary

- `cooperation_rate`: fraction of person agents that are not defecting.
- `free_rider_ratio`: fraction of person agents that are defecting.
- `resource_utilization`: fraction of resources currently occupied.
- `gini_coefficient`: inequality in cumulative resource usage (0 = perfect equality).
- `sustainability_index`: cumulative ratio of idle resource ticks to total resource ticks.
- `conflict_rate`: cumulative conflicts divided by elapsed model steps.
- `sanction_rate`: cumulative sanctions divided by elapsed model steps.
- `iforest_anomaly_ratio`: latest anomaly count divided by number of person agents.
- `mean_trust`, `mean_autonomy`, `mean_satisfaction`: population means over person agents.

## Governance Modes

The runtime `system_type` values are:
- `baseline`
- `ai_advisory`
- `ai_autonomous`
- `blockchain_partial`
- `blockchain_full`
- `integrated`

Note on naming:
- The experiment matrix has 8 **config names**, but only 6 distinct `system_type` values.
- Three integrated config variants (`integrated_ap`, `integrated_af`, `integrated_x`) all map to `system_type = integrated` with different `procedural_bonus_modifier` values.

## Model Execution Loop

At each simulation step:

1. Agent activation order is shuffled.
2. `PersonAgent` either requests a resource or continues using one.
3. If blocked/unavailable, waiting and frustration dynamics are updated.
4. On resource release, trust/satisfaction/autonomy updates are applied.
5. Every `detect_every_n_steps`, Isolation Forest detection runs.
6. For selected modes, anomalies can trigger temporary request blocks and sanctions.
7. `DataCollector` records model and agent outputs.

## Reproducibility Notes

In batch runs, each run uses:

- `seed = scenario_id * 1000 + repeat_idx`

This improves reproducibility across reruns with the same code and dependency versions. Full reproducibility still depends on strict environment/version control.

## Performance Notes

- Full batch runs (2600 simulations) can be time-consuming.
- Start with reduced runs (`max_runs`) for sanity checks.
- Output CSVs can grow large; monitor disk usage.

## Troubleshooting

### `ImportError: No module named ...`

Install required dependencies:

```bash
pip install mesa solara pandas scikit-learn statsmodels
```

### `FileNotFoundError` in `analysis_report.py`

Generate batch outputs first:

```bash
python batch_run.py
python analysis_report.py
```

### Long runtime

- Use a reduced run size for testing.
- Use `max_runs` for early termination smoke tests.

## Development Notes

- Core model: `model/model.py`
- Agent behavior: `agents/person.py`, `agents/resource.py`
- Psychology functions: `modules/psychology.py`
- Interactive app entry: `run.py`
- Batch pipeline: `batch_run.py`
- Statistical report pipeline: `analysis_report.py`

The current structure is modular enough to add:
- new agent archetypes
- additional governance policies
- alternative sanction/detection mechanisms

## Roadmap Ideas

- Parameter calibration and sensitivity analysis
- Trajectory-aware statistics (beyond final-tick analysis)
- Alternative anomaly detection benchmarks
- RL or policy-search based governance optimization
- External validation with empirical datasets