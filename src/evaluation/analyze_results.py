"""
Statistical Analysis and Visualization for Multi-Turn Sycophancy Experiments.

Generates publication-quality figures:
1. Confidence Trajectory vs. Turn Step t (with shaded standard error)
2. Confidence vs. Cumulative Social Pressure S_t (LOWESS regression phase plot)
3. Mixed-Effects Regression summary isolating turn position effect from pressure effect.
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


def analyze_and_plot(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} records from {csv_path}")

    # Set aesthetics
    sns.set_theme(style="whitegrid", font_scale=1.15)
    palette = {"positive": "#1b9e77", "negative": "#d95f02"}

    # --------------------------------------------------------------------------
    # Figure 1: Confidence vs. Conversational Turn t
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="turn",
        y="target_token_confidence",
        hue="track",
        style="track",
        markers=True,
        dashes=False,
        palette=palette,
        errorbar="se",
        ax=ax,
    )
    ax.set_title("Target-Token Softmax Confidence vs. Turn (t)", fontweight="bold")
    ax.set_xlabel("Conversational Turn t")
    ax.set_ylabel("P(Valid) (%)")
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "confidence_vs_turn.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Figure 1 saved -> {fig1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: Confidence vs. Cumulative Pressure Score S_t
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="running_prompt_score",
        y="target_token_confidence",
        hue="track",
        alpha=0.35,
        palette=palette,
        legend=False,
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="running_prompt_score",
        y="target_token_confidence",
        scatter=False,
        color="#2b5c8f",
        lowess=True,
        ax=ax,
    )
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7, label="Neutral (Turn 0)")
    ax.set_title("Confidence vs. Cumulative Pressure Score (S_t)", fontweight="bold")
    ax.set_xlabel("Cumulative Social Pressure (S_t)")
    ax.set_ylabel("P(Valid) (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "confidence_vs_cum_pressure.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Figure 2 saved -> {fig2_path}")

    # --------------------------------------------------------------------------
    # Statistical Modeling: Mixed-Effects Linear Model (LMM)
    # --------------------------------------------------------------------------
    # Tests whether cumulative pressure S_t has a significant effect independent of turn position
    summary_path = os.path.join(output_dir, "regression_summary.txt")
    try:
        model = smf.mixedlm(
            "target_token_confidence ~ turn + running_prompt_score",
            df,
            groups=df["claim_id"],
        )
        fit_result = model.fit()
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(fit_result.summary().as_text())
        print(f"Mixed-effects regression summary saved -> {summary_path}")
        print("\n--- Regression Coefficients ---")
        print(fit_result.summary().tables[1])
    except Exception as e:
        print(f"Could not fit mixed-effects model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True, help="Path to claims_drift_summary.csv")
    parser.add_argument("--output_dir", type=str, default="results/figures", help="Output directory for plots")
    args = parser.parse_args()

    analyze_and_plot(args.results_csv, args.output_dir)
