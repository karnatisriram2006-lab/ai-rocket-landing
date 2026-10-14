
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Baseline": RESULTS_DIR / "evaluation_ppo_baseline.csv",
    "Shaped": RESULTS_DIR / "evaluation_ppo_shaped.csv",
    "Reward hacking": RESULTS_DIR / "evaluation_ppo_hack.csv",
}


def read_boolean_column(series):
    """Convert CSV boolean values safely."""
    values = series.astype(str).str.strip().str.lower()
    converted = values.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })

    if converted.isna().any():
        raise ValueError(
            f"Unexpected boolean values: {series.unique()}"
        )

    return converted.astype(bool)


def main():
    summaries = []
    reward_distributions = []
    model_names = []

    for model_name, csv_path in MODEL_FILES.items():
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run evaluate.py first."
            )

        df = pd.read_csv(csv_path)

        required = {
            "reward",
            "score_at_least_200",
            "landed",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {missing}"
            )

        rewards = pd.to_numeric(
            df["reward"], errors="raise"
        )

        score_success = read_boolean_column(
            df["score_at_least_200"]
        )
        landing_success = read_boolean_column(
            df["landed"]
        )

        episode_count = len(df)

        if episode_count == 0:
            raise ValueError(f"{csv_path} contains no episodes.")

        summaries.append({
            "model": model_name,
            "episodes": episode_count,
            "mean_reward": rewards.mean(),
            "reward_std": rewards.std(ddof=0),
            "episodes_score_ge_200": int(score_success.sum()),
            "score_rate_percent": 100 * score_success.mean(),
            "detected_landings": int(landing_success.sum()),
            "landing_rate_percent": 100 * landing_success.mean(),
        })

        model_names.append(model_name)
        reward_distributions.append(rewards.to_numpy())

    summary = pd.DataFrame(summaries)

    summary_path = RESULTS_DIR / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(summary.round(2).to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")

    # Graph 1: Mean reward under the original environment reward.
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        summary["model"],
        summary["mean_reward"],
    )
    plt.axhline(200, linestyle="--", label="200-point benchmark")
    plt.bar_label(bars, fmt="%.1f", padding=3)
    plt.ylabel("Mean evaluation reward")
    plt.title("Lunar Lander: Mean Reward Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mean_reward.png", dpi=200)
    plt.close()

    # Graph 2: Score-threshold rate vs detected landing rate.
    x = np.arange(len(summary))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - width / 2,
        summary["score_rate_percent"],
        width,
        label="Episodes scoring >= 200",
    )
    plt.bar(
        x + width / 2,
        summary["landing_rate_percent"],
        width,
        label="Detected landing rate",
    )

    plt.xticks(x, summary["model"])
    plt.ylabel("Episodes (%)")
    plt.ylim(0, 100)
    plt.title("Score Threshold vs Landing Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "success_rates.png", dpi=200)
    plt.close()

    # Graph 3: Distribution of episode rewards.
    plt.figure(figsize=(9, 5))
    plt.boxplot(
        reward_distributions,
        tick_labels=model_names,
        showmeans=True,
    )
    plt.axhline(200, linestyle="--", label="200-point benchmark")
    plt.ylabel("Episode reward")
    plt.title("Evaluation Reward Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reward_distribution.png", dpi=200)
    plt.close()

    print(f"Graphs saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
