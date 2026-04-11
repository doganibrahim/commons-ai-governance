import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

MODEL_PATH = "output/model_metrics_104x25.csv"
SCENARIO_PATH = "output/scenario_matrix_104.csv"
REPORT_PATH = "output/academic_report_summary.md"


def final_tick(df):
    return (
        df.sort_values(["scenario_id", "repeat_id", "tick"])
        .groupby(["scenario_id", "repeat_id"], as_index=False)
        .tail(1)
        .copy()
    )


def run_anova(final_df, metric):
    model = ols(
        f"{metric} ~ C(system_type) + C(cohesion) + N_people + N_resources",
        data=final_df,
    ).fit()
    try:
        anova = sm.stats.anova_lm(model, typ=2)
    except Exception:
        anova = sm.stats.anova_lm(model, typ=1)
    ss_total = anova["sum_sq"].sum()
    anova = anova.copy()
    anova["eta_sq"] = anova["sum_sq"] / ss_total
    if final_df["system_type"].nunique() >= 2:
        tukey = pairwise_tukeyhsd(
            endog=final_df[metric], groups=final_df["system_type"], alpha=0.05
        )
    else:
        tukey = "Tukey skipped: only one system_type present in data."
    return anova, tukey


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run batch experiment first (python batch_run.py)."
        )
    if not os.path.exists(SCENARIO_PATH):
        raise FileNotFoundError(
            f"Missing {SCENARIO_PATH}. Run batch experiment first (python batch_run.py)."
        )

    df = pd.read_csv(MODEL_PATH)
    scenarios = pd.read_csv(SCENARIO_PATH)
    fdf = final_tick(df)

    metrics = [
        "gini_coefficient",
        "mean_trust",
        "resource_utilization",
        "free_rider_ratio",
    ]

    results = {}
    for m in metrics:
        results[m] = run_anova(fdf, m)

    lines = []
    lines.append("# Academic Report Summary")
    lines.append("")
    lines.append("## Data Status")
    lines.append(f"- number of rows in scenario matrix: {len(scenarios)}")
    lines.append(f"- number of rows in model metrics: {len(df)}")
    lines.append(f"- number of final tick runs: {len(fdf)}")
    lines.append("")
    lines.append("## Hypothesis-Based Short Summary")
    lines.append("- H1: System type fairness metrics (Gini) significantly affect.")
    lines.append("- H2: System type psychological trust levels significantly affect.")
    lines.append("- H3: System type resource utilization and free-rider ratio significantly affect.")
    lines.append("")
    lines.append("## ANOVA Summary (system_type effect)")

    summary_rows = []
    for metric in metrics:
        anova, _ = results[metric]
        if "C(system_type)" in anova.index:
            pval = anova.loc["C(system_type)", "PR(>F)"]
            eta = anova.loc["C(system_type)", "eta_sq"]
            summary_rows.append((metric, pval, eta))
            lines.append(f"- {metric}: p={pval:.4g}, eta^2={eta:.4f}")
        else:
            lines.append(f"- {metric}: C(system_type) row not found")

    lines.append("")
    lines.append("## Most Critical 3 Findings")
    summary_rows.sort(key=lambda x: x[2], reverse=True)
    for metric, pval, eta in summary_rows[:3]:
        significance = "significant" if pval < 0.05 else "not significant"
        lines.append(
            f"- {metric}: system_type etkisi {significance}; etki buyuklugu eta^2={eta:.4f}."
        )

    lines.append("")
    lines.append("## Tukey HSD Notes")
    lines.append("- The following tables show pairwise system_type comparisons.")
    for metric in metrics:
        _, tukey = results[metric]
        lines.append("")
        lines.append(f"### Tukey: {metric}")
        lines.append("```")
        lines.append(str(tukey))
        lines.append("```")

    lines.append("")
    lines.append("## Validity / Limitations Notes")
    lines.append("- This summary is calculated based on final-tick; time series effects should be additionally analyzed.")
    lines.append("- The reliability of the results is related to the number of repetitions and scenario coverage.")
    lines.append("- Parameter calibration (especially procedural bonus, sanction windows) should be reported additionally for validity.")

    os.makedirs("output", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
