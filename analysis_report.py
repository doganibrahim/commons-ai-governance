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
    lines.append("# Akademik Rapor Ozeti")
    lines.append("")
    lines.append("## Veri Durumu")
    lines.append(f"- Scenario matrix satir sayisi: {len(scenarios)}")
    lines.append(f"- Model metric satir sayisi: {len(df)}")
    lines.append(f"- Final tick run sayisi: {len(fdf)}")
    lines.append("")
    lines.append("## Hipotez Bazli Kisa Yorum")
    lines.append("- H1: Sistem tipi adalet metriklerini (Gini) anlamli bicimde etkiler.")
    lines.append("- H2: Sistem tipi psikolojik guven seviyelerini anlamli bicimde etkiler.")
    lines.append("- H3: Sistem tipi kaynak verimliligi ve free-rider oranlarini anlamli bicimde etkiler.")
    lines.append("")
    lines.append("## ANOVA Ozeti (system_type etkisi)")

    summary_rows = []
    for metric in metrics:
        anova, _ = results[metric]
        if "C(system_type)" in anova.index:
            pval = anova.loc["C(system_type)", "PR(>F)"]
            eta = anova.loc["C(system_type)", "eta_sq"]
            summary_rows.append((metric, pval, eta))
            lines.append(f"- {metric}: p={pval:.4g}, eta^2={eta:.4f}")
        else:
            lines.append(f"- {metric}: C(system_type) satiri bulunamadi")

    lines.append("")
    lines.append("## En Kritik 3 Bulgu")
    summary_rows.sort(key=lambda x: x[2], reverse=True)
    for metric, pval, eta in summary_rows[:3]:
        significance = "anlamli" if pval < 0.05 else "anlamli degil"
        lines.append(
            f"- {metric}: system_type etkisi {significance}; etki buyuklugu eta^2={eta:.4f}."
        )

    lines.append("")
    lines.append("## Tukey HSD Notlari")
    lines.append("- Asagidaki tablolar pairwise system_type karsilastirmalarini verir.")
    for metric in metrics:
        _, tukey = results[metric]
        lines.append("")
        lines.append(f"### Tukey: {metric}")
        lines.append("```")
        lines.append(str(tukey))
        lines.append("```")

    lines.append("")
    lines.append("## Gecerlilik / Sinirlilik Notlari")
    lines.append("- Bu ozet final-tick uzerinden hesaplanmistir; zaman serisi etkileri ayrica incelenmelidir.")
    lines.append("- Ciktilarin guvenilirligi tekrar sayisi ve senaryo kapsamina baglidir.")
    lines.append("- Parametre kalibrasyonu (ozellikle procedural bonus, sanction windows) ic gecerlilik icin ayrica raporlanmalidir.")

    os.makedirs("output", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
