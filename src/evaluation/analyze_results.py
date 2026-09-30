"""
Statistical Analysis and Visualization for Multi-Turn Sycophancy Experiments (O1 Trajectories).

Generates publication-quality figures:
1. Confidence Trajectory vs. Turn Step t (with confidence floor phi = 50% line and shaded SE)
2. Ground-Truth Accuracy Trajectory (corr_t) vs. Turn Step t
3. Confidence vs. Cumulative Social Pressure S_t (LOWESS regression phase plot)
4. Trajectory Stratification by Claim Difficulty (Hard vs. Advanced)
5. Mixed-Effects Regression summary isolating turn position effect from pressure effect.
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
    # Figure 1: Confidence Trajectory vs. Conversational Turn t
    # --------------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=df,
            x="turn",
            y="target_token_confidence",
            hue="track" if "track" in df.columns else None,
            style="track" if "track" in df.columns else None,
            markers=True,
            dashes=False,
            palette=palette if "track" in df.columns else None,
            errorbar="se",
            ax=ax,
        )
        ax.axhline(50.0, color="#d95f02", linestyle=":", alpha=0.8, label="Confidence Floor (\u03c6 = 50%)")
        ax.set_title("Target-Token Softmax Confidence vs. Turn (t)", fontweight="bold")
        ax.set_xlabel("Conversational Turn t")
        ax.set_ylabel("P(Valid) (%)")
        ax.set_ylim(-5, 105)
        ax.legend(loc="lower left")
        plt.tight_layout()
        fig1_path = os.path.join(output_dir, "confidence_vs_turn.png")
        plt.savefig(fig1_path, dpi=300)
        plt.close()
        print(f"Figure 1 saved -> {fig1_path}")
    except Exception as e:
        print(f"Warning: Could not generate Figure 1: {e}")

    # --------------------------------------------------------------------------
    # Figure 2: Ground-Truth Accuracy (corr_t) vs. Conversational Turn t (O1 Core)
    # --------------------------------------------------------------------------
    try:
        if "corr_t" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(
                data=df,
                x="turn",
                y="corr_t",
                hue="track" if "track" in df.columns else None,
                style="track" if "track" in df.columns else None,
                markers=True,
                dashes=False,
                palette=palette if "track" in df.columns else None,
                errorbar="se",
                ax=ax,
            )
            ax.set_title("Ground-Truth Accuracy Trajectory (corr_t) vs. Turn (t)", fontweight="bold")
            ax.set_xlabel("Conversational Turn t")
            ax.set_ylabel("Accuracy corr_t \u2208 {0, 1}")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="lower left")
            plt.tight_layout()
            fig2_path = os.path.join(output_dir, "correctness_vs_turn.png")
            plt.savefig(fig2_path, dpi=300)
            plt.close()
            print(f"Figure 2 (Ground-Truth Accuracy Trajectory) saved -> {fig2_path}")
    except Exception as e:
        print(f"Warning: Could not generate Figure 2: {e}")

    # --------------------------------------------------------------------------
    # Figure 3: Confidence vs. Cumulative Pressure Score S_t
    # --------------------------------------------------------------------------
    try:
        score_col = "running_prompt_score" if "running_prompt_score" in df.columns else "aggregate_prompt_score"
        if score_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                data=df,
                x=score_col,
                y="target_token_confidence",
                hue="track" if "track" in df.columns else None,
                alpha=0.35,
                palette=palette if "track" in df.columns else None,
                legend=False,
                ax=ax,
            )
            sns.regplot(
                data=df,
                x=score_col,
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
            fig3_path = os.path.join(output_dir, "confidence_vs_cum_pressure.png")
            plt.savefig(fig3_path, dpi=300)
            plt.close()
            print(f"Figure 3 saved -> {fig3_path}")
    except Exception as e:
        print(f"Warning: Could not generate Figure 3: {e}")

    # --------------------------------------------------------------------------
    # Figure 4: Stratified Trajectory by Claim Difficulty
    # --------------------------------------------------------------------------
    try:
        if "difficulty" in df.columns and "corr_t" in df.columns:
            fig, ax = plt.subplots(figsize=(8.5, 5))
            sns.lineplot(
                data=df,
                x="turn",
                y="corr_t",
                hue="difficulty",
                style="difficulty",
                markers=True,
                dashes=False,
                palette="Set2",
                errorbar="se",
                ax=ax,
            )
            ax.set_title("Ground-Truth Accuracy Trajectory Stratified by Claim Difficulty", fontweight="bold")
            ax.set_xlabel("Conversational Turn t")
            ax.set_ylabel("Accuracy corr_t \u2208 {0, 1}")
            ax.set_ylim(-0.05, 1.05)
            plt.tight_layout()
            fig4_path = os.path.join(output_dir, "accuracy_vs_difficulty.png")
            plt.savefig(fig4_path, dpi=300)
            plt.close()
            print(f"Figure 4 saved -> {fig4_path}")
    except Exception as e:
        print(f"Warning: Could not generate Figure 4: {e}")

    # --------------------------------------------------------------------------
    # Statistical Modeling: Mixed-Effects Linear Model (LMM)
    # --------------------------------------------------------------------------
    summary_path = os.path.join(output_dir, "regression_summary.txt")
    try:
        score_col = "running_prompt_score" if "running_prompt_score" in df.columns else "aggregate_prompt_score"
        model = smf.mixedlm(
            f"target_token_confidence ~ turn + {score_col}",
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

