"""
Tek bir senaryo çalıştırıp DataCollector çıktılarını CSV'ye yazan
basit batch runner. İleride 2.600 koşuluk deney matrisine genişletilecek.

Kullanım:
    python batch_run.py
"""

import os
from model.model import CommonsModel

PARAMS = {
    "N_people": 20,
    "N_resources": 8,
    "width": 15,
    "height": 15,
    "system_type": "baseline",
    "procedural_bonus_modifier": 0.0,
}

N_STEPS = 300
OUTPUT_DIR = "output"


def run_single(params: dict, n_steps: int, run_id: int = 0):
    model = CommonsModel(**params)
    for _ in range(n_steps):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    model_df["run_id"] = run_id
    model_df["system_type"] = params["system_type"]

    agent_df = model.datacollector.get_agent_vars_dataframe()
    agent_df["run_id"] = run_id
    agent_df["system_type"] = params["system_type"]

    return model_df, agent_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Simulasyon basliyor: {N_STEPS} adim, parametreler: {PARAMS}")
    model_df, agent_df = run_single(PARAMS, N_STEPS)

    model_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    agent_path = os.path.join(OUTPUT_DIR, "agent_metrics.csv")

    model_df.to_csv(model_path)
    agent_df.to_csv(agent_path)

    print(f"Model metrikleri -> {model_path}  ({len(model_df)} satir)")
    print(f"Ajan metrikleri  -> {agent_path}  ({len(agent_df)} satir)")
    print()
    print("Son 5 adim model metrikleri:")
    print(model_df.tail().to_string())


if __name__ == "__main__":
    main()
