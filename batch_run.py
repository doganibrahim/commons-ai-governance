"""Batch experiment runner: 104 scenarios x 25 repeats x 300 steps."""

import os
from model.model import CommonsModel

N_STEPS = 300
N_REPEATS = 25
GRID_WIDTH = 15
GRID_HEIGHT = 15
OUTPUT_DIR = "output"
MODEL_OUT = os.path.join(OUTPUT_DIR, "model_metrics_104x25.csv")
AGENT_OUT = os.path.join(OUTPUT_DIR, "agent_metrics_104x25.csv")


def build_system_configs():
    return [
        {"config_name": "baseline", "system_type": "baseline", "procedural_bonus_modifier": 0.00},
        {"config_name": "ai_advisory", "system_type": "ai_advisory", "procedural_bonus_modifier": 0.02},
        {"config_name": "ai_autonomous", "system_type": "ai_autonomous", "procedural_bonus_modifier": -0.03},
        {"config_name": "blockchain_partial", "system_type": "blockchain_partial", "procedural_bonus_modifier": 0.03},
        {"config_name": "blockchain_full", "system_type": "blockchain_full", "procedural_bonus_modifier": 0.06},
        {"config_name": "integrated_ap", "system_type": "integrated", "procedural_bonus_modifier": 0.05},
        {"config_name": "integrated_af", "system_type": "integrated", "procedural_bonus_modifier": 0.07},
        {"config_name": "integrated_x", "system_type": "integrated", "procedural_bonus_modifier": 0.04},
    ]


def build_context_profiles():
    # 13 profile x 8 systems = 104 scenarios
    return [
        {"context_id": "c01", "N_people": 18, "N_resources": 8, "base_trust_shift": -5, "cohesion": "low", "agent_type_distribution": {"ideal": 0.15, "standard": 0.55, "toxic": 0.30}},
        {"context_id": "c02", "N_people": 18, "N_resources": 8, "base_trust_shift": 0, "cohesion": "mid", "agent_type_distribution": {"ideal": 0.25, "standard": 0.50, "toxic": 0.25}},
        {"context_id": "c03", "N_people": 18, "N_resources": 8, "base_trust_shift": 5, "cohesion": "high", "agent_type_distribution": {"ideal": 0.35, "standard": 0.50, "toxic": 0.15}},
        {"context_id": "c04", "N_people": 24, "N_resources": 8, "base_trust_shift": -5, "cohesion": "low", "agent_type_distribution": {"ideal": 0.15, "standard": 0.50, "toxic": 0.35}},
        {"context_id": "c05", "N_people": 24, "N_resources": 8, "base_trust_shift": 0, "cohesion": "mid", "agent_type_distribution": {"ideal": 0.25, "standard": 0.50, "toxic": 0.25}},
        {"context_id": "c06", "N_people": 24, "N_resources": 8, "base_trust_shift": 5, "cohesion": "high", "agent_type_distribution": {"ideal": 0.40, "standard": 0.45, "toxic": 0.15}},
        {"context_id": "c07", "N_people": 30, "N_resources": 8, "base_trust_shift": -5, "cohesion": "low", "agent_type_distribution": {"ideal": 0.10, "standard": 0.50, "toxic": 0.40}},
        {"context_id": "c08", "N_people": 30, "N_resources": 8, "base_trust_shift": 0, "cohesion": "mid", "agent_type_distribution": {"ideal": 0.20, "standard": 0.55, "toxic": 0.25}},
        {"context_id": "c09", "N_people": 30, "N_resources": 8, "base_trust_shift": 5, "cohesion": "high", "agent_type_distribution": {"ideal": 0.35, "standard": 0.50, "toxic": 0.15}},
        {"context_id": "c10", "N_people": 24, "N_resources": 10, "base_trust_shift": -3, "cohesion": "low", "agent_type_distribution": {"ideal": 0.20, "standard": 0.50, "toxic": 0.30}},
        {"context_id": "c11", "N_people": 24, "N_resources": 10, "base_trust_shift": 0, "cohesion": "mid", "agent_type_distribution": {"ideal": 0.30, "standard": 0.50, "toxic": 0.20}},
        {"context_id": "c12", "N_people": 24, "N_resources": 10, "base_trust_shift": 3, "cohesion": "high", "agent_type_distribution": {"ideal": 0.40, "standard": 0.45, "toxic": 0.15}},
        {"context_id": "c13", "N_people": 20, "N_resources": 7, "base_trust_shift": 0, "cohesion": "mid", "agent_type_distribution": {"ideal": 0.25, "standard": 0.50, "toxic": 0.25}},
    ]


def build_scenarios():
    scenarios = []
    sid = 1
    for cfg in build_system_configs():
        for ctx in build_context_profiles():
            scenarios.append(
                {
                    "scenario_id": sid,
                    "config_name": cfg["config_name"],
                    "system_type": cfg["system_type"],
                    "procedural_bonus_modifier": cfg["procedural_bonus_modifier"],
                    "context_id": ctx["context_id"],
                    "N_people": ctx["N_people"],
                    "N_resources": ctx["N_resources"],
                    "base_trust_shift": ctx["base_trust_shift"],
                    "cohesion": ctx["cohesion"],
                    "agent_type_distribution": ctx["agent_type_distribution"],
                }
            )
            sid += 1
    return scenarios


def run_single_scenario(scenario, repeat_idx, n_steps):
    seed = scenario["scenario_id"] * 1000 + repeat_idx
    model = CommonsModel(
        N_people=scenario["N_people"],
        N_resources=scenario["N_resources"],
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        system_type=scenario["system_type"],
        procedural_bonus_modifier=scenario["procedural_bonus_modifier"],
        agent_type_distribution=scenario["agent_type_distribution"],
        random_seed=seed,
    )
    for _ in range(n_steps):
        model.step()
    model_df = model.datacollector.get_model_vars_dataframe().reset_index(names="tick")
    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    for df in (model_df, agent_df):
        df["scenario_id"] = scenario["scenario_id"]
        df["repeat_id"] = repeat_idx
        df["config_name"] = scenario["config_name"]
        df["system_type"] = scenario["system_type"]
        df["context_id"] = scenario["context_id"]
        df["N_people"] = scenario["N_people"]
        df["N_resources"] = scenario["N_resources"]
        df["cohesion"] = scenario["cohesion"]
        df["base_trust_shift"] = scenario["base_trust_shift"]
    return model_df, agent_df


def append_csv(df, path):
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=write_header)


def run_experiments(max_runs=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for p in (MODEL_OUT, AGENT_OUT):
        if os.path.exists(p):
            os.remove(p)
    scenarios = build_scenarios()
    total_runs = len(scenarios) * N_REPEATS
    print(f"Running {len(scenarios)} scenarios x {N_REPEATS} repeats = {total_runs} runs")
    run_counter = 0
    for scenario in scenarios:
        for rep in range(1, N_REPEATS + 1):
            run_counter += 1
            mdf, adf = run_single_scenario(scenario, rep, N_STEPS)
            append_csv(mdf, MODEL_OUT)
            append_csv(adf, AGENT_OUT)
            if run_counter % 20 == 0:
                print(f"Progress: {run_counter}/{total_runs} runs")
            if max_runs is not None and run_counter >= max_runs:
                print("Stopped early due to max_runs limit.")
                return
    print("All runs complete.")
    print(f"Model CSV: {MODEL_OUT}")
    print(f"Agent CSV: {AGENT_OUT}")


if __name__ == "__main__":
    # For quick smoke test use: run_experiments(max_runs=2)
    run_experiments()
