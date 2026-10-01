"""
Statistical Analysis and Visualization Suite for Reversibility & Oscillation Experiment.

Analyzes multi-turn reversibility and oscillation under sustained criticism:
1. Flip Rate over Turns (Cumulative and Instantaneous Hazard)
2. Reversibility Outcome Distribution (Settled Wrong vs. Swung Back vs. Oscillated vs. Resilient)
3. Oscillation / Flip-Count Distribution
4. Trajectory Confidence Dynamics Stratified by Reversibility Pattern
5. Statistical Significance Tests Across Claim Difficulty (Hard vs. Advanced)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def analyze_and_plot_reversibility(
    summary_csv: str,
    claim_summary_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Generates all diagnostic figures, summary tables, and statistical tests for reversibility.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(summary_csv), "figures")
    os.makedirs(output_dir, exist_ok=True)

    df_turns = pd.read_csv(summary_csv)
    if claim_summary_csv and os.path.exists(claim_summary_csv):
        df_claims = pd.read_csv(claim_summary_csv)
    else:
        # Reconstruct claim-level summary from turns if not provided
        df_claims = reconstruct_claim_summary(df_turns)

    print(f"Loaded {len(df_turns)} turn records and {len(df_claims)} claim trajectories.")

    # Aesthetic styling
    sns.set_theme(style="whitegrid", font_scale=1.15)
    palette_diff = {"Hard": "#386cb0", "Advanced": "#f0027f", "Overall": "#333333"}
    palette_outcomes = {
        "settled_wrong": "#d95f02",       # Red-orange (collapse)
        "swung_back_correct": "#1b9e77",  # Green (recovery)
        "oscillated": "#7570b3",          # Purple (oscillation)
        "resilient_correct": "#2b5c8f",   # Blue (invariant)
    }

    # --------------------------------------------------------------------------
    # Figure 1: Flip Rate Trajectory over Turns (Cumulative & Instantaneous)
    # --------------------------------------------------------------------------
    try:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.2]}
        )

        # Calculate turn-by-turn cumulative flip rates per difficulty and overall
        max_turn = int(df_turns["turn"].max())
        turn_range = list(range(1, max_turn + 1))

        curve_data = []
        for t in turn_range:
            # Overall
            flipped_by_t_all = df_turns[(df_turns["turn"] <= t) & (df_turns["is_flip"] == True)]["claim_id"].nunique()
            active_t_all = len(df_turns[df_turns["turn"] == t])
            flips_at_t_all = (df_turns[df_turns["turn"] == t]["is_flip"] == True).sum()
            total_claims_all = df_claims["claim_id"].nunique()

            curve_data.append({
                "turn": t,
                "difficulty": "Overall",
                "cum_flip_rate": (flipped_by_t_all / total_claims_all * 100.0) if total_claims_all > 0 else 0.0,
                "instantaneous_flip_rate": (flips_at_t_all / active_t_all * 100.0) if active_t_all > 0 else 0.0,
            })

            # By difficulty
            for diff in ["Hard", "Advanced"]:
                diff_claims = df_claims[df_claims["difficulty"] == diff]["claim_id"].unique()
                if len(diff_claims) == 0:
                    continue
                diff_turns = df_turns[df_turns["claim_id"].isin(diff_claims)]
                flipped_by_t = diff_turns[(diff_turns["turn"] <= t) & (diff_turns["is_flip"] == True)]["claim_id"].nunique()
                active_t = len(diff_turns[diff_turns["turn"] == t])
                flips_at_t = (diff_turns[diff_turns["turn"] == t]["is_flip"] == True).sum()

                curve_data.append({
                    "turn": t,
                    "difficulty": diff,
                    "cum_flip_rate": (flipped_by_t / len(diff_claims) * 100.0),
                    "instantaneous_flip_rate": (flips_at_t / active_t * 100.0) if active_t > 0 else 0.0,
                })

        df_curves = pd.DataFrame(curve_data)

        # Top Panel: Cumulative Flip Rate Curve
        for diff in ["Overall", "Hard", "Advanced"]:
            sub = df_curves[df_curves["difficulty"] == diff]
            if len(sub) == 0:
                continue
            lw = 3.0 if diff == "Overall" else 2.0
            ls = "-" if diff != "Overall" else "--"
            marker = "o" if diff == "Overall" else "s"
            ax_top.plot(
                sub["turn"],
                sub["cum_flip_rate"],
                label=f"{diff} Claims",
                color=palette_diff.get(diff, "#555555"),
                linewidth=lw,
                linestyle=ls,
                marker=marker,
                markersize=6,
            )

        ax_top.set_title("Reversibility: Cumulative Flip Rate over Criticism Turns", fontweight="bold")
        ax_top.set_ylabel("Cumulative Flip Rate (%)")
        ax_top.set_ylim(-2, 102)
        ax_top.legend(loc="upper left")

        # Bottom Panel: Instantaneous Flip Rate
        bar_width = 0.25
        turns_arr = np.array(turn_range)
        diffs = [d for d in ["Hard", "Advanced"] if d in df_curves["difficulty"].values]
        for i, diff in enumerate(diffs):
            sub = df_curves[df_curves["difficulty"] == diff]
            offset = (i - 0.5) * bar_width
            ax_bot.bar(
                turns_arr + offset,
                sub["instantaneous_flip_rate"],
                width=bar_width,
                label=f"{diff} (Instantaneous)",
                color=palette_diff.get(diff, "#555555"),
                alpha=0.8,
            )

        ax_bot.set_title("Instantaneous Flip Likelihood at Turn t", fontweight="bold")
        ax_bot.set_xlabel("Sustained Criticism Turn t (1 .. T)")
        ax_bot.set_ylabel("Flip Hazard (%)")
        ax_bot.set_xticks(turn_range)
        ax_bot.set_ylim(0, max(df_curves["instantaneous_flip_rate"].max() * 1.25, 10))
        ax_bot.legend(loc="upper right")

        plt.tight_layout()
        fig1_path = os.path.join(output_dir, "flip_rate_over_turns.png")
        plt.savefig(fig1_path, dpi=300)
        plt.close()
        print(f"Figure 1 saved -> {fig1_path}")
    except Exception as e:
        print(f"Warning: Could not generate Figure 1: {e}")

    # --------------------------------------------------------------------------
    # Figure 2: Reversibility Outcome Proportions (Stacked Bar Chart)
    # --------------------------------------------------------------------------
    try:
        outcome_order = ["settled_wrong", "swung_back_correct", "oscillated", "resilient_correct"]
        outcome_labels = {
            "settled_wrong": "Settled Permanently on Wrong",
            "swung_back_correct": "Swung Back to Correct on Its Own",
            "oscillated": "Oscillated (>= 2 Flips)",
            "resilient_correct": "Resilient (Never Flipped)",
        }

        # Calculate proportions overall and by difficulty
        records = []
        for group_name in ["Overall", "Hard", "Advanced"]:
            if group_name == "Overall":
                sub = df_claims[df_claims["corr_t0"] == 1]
            else:
                sub = df_claims[(df_claims["difficulty"] == group_name) & (df_claims["corr_t0"] == 1)]

            total = len(sub)
            if total == 0:
                continue

            counts = sub["reversibility_category"].value_counts().to_dict()
            for cat in outcome_order:
                cnt = counts.get(cat, 0)
                records.append({
                    "group": group_name,
                    "category": cat,
                    "count": cnt,
                    "percentage": cnt / total * 100.0,
                })

        df_outcomes = pd.DataFrame(records)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        groups = [g for g in ["Overall", "Hard", "Advanced"] if g in df_outcomes["group"].values]
        bottoms = np.zeros(len(groups))

        for cat in outcome_order:
            sub = df_outcomes[df_outcomes["category"] == cat]
            sub_pcts = [sub[sub["group"] == g]["percentage"].values[0] if len(sub[sub["group"] == g]) > 0 else 0.0 for g in groups]
            bars = ax.bar(
                groups,
                sub_pcts,
                bottom=bottoms,
                label=outcome_labels[cat],
                color=palette_outcomes[cat],
                edgecolor="white",
                linewidth=1.2,
                width=0.55,
            )
            # Add text labels on segments >= 6%
            for idx, (p, b) in enumerate(zip(sub_pcts, bottoms)):
                if p >= 6.0:
                    ax.text(
                        idx,
                        b + p / 2.0,
                        f"{p:.1f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=10,
                    )
            bottoms += np.array(sub_pcts)

        ax.set_title("Reversibility: Model Behavior Under Sustained Criticism Following Initial Correctness", fontweight="bold")
        ax.set_ylabel("Proportion of Initially Correct Claims (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
        plt.tight_layout()
        fig2_path = os.path.join(output_dir, "reversibility_outcomes.png")
        plt.savefig(fig2_path, dpi=300)
        plt.close()
        print(f"Figure 2 saved -> {fig2_path}")
    except Exception as e:
        print(f"Warning: Could not generate Figure 2: {e}")

    # --------------------------------------------------------------------------
    # Figure 3: Oscillation / Flip Count Distribution
    # --------------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sns.countplot(
            data=df_claims[df_claims["corr_t0"] == 1],
            x="total_flips",
            hue="difficulty",
            palette={"Hard": "#386cb0", "Advanced": "#f0027f"},
            ax=ax,
        )
        ax.set_title("Reversibility: Distribution of Flip Counts Under Sustained Criticism", fontweight="bold")
        ax.set_xlabel("Total Flips per Claim (0 = Resilient, 1 = Settled Flip, >=2 = Oscillation)")
        ax.set_ylabel("Number of Claims")
        ax.legend(title="Claim Difficulty")
        plt.tight_layout()
        fig3_path = os.path.join(output_dir, "oscillation_distribution.png")
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        print(f"Figure 3 saved -> {fig3_path}")
    except Exception as e:
        print(f"Warning: Could not generate Figure 3: {e}")

    # --------------------------------------------------------------------------
    # Figure 4: Confidence Trajectories by Reversibility Pattern
    # --------------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        if "reversibility_category" not in df_turns.columns:
            # Map category from df_claims
            cat_map = df_claims.set_index("claim_id")["reversibility_category"].to_dict()
            df_turns["reversibility_category"] = df_turns["claim_id"].map(cat_map)

        sub_turns = df_turns[df_turns["reversibility_category"].isin(palette_outcomes.keys())]
        sns.lineplot(
            data=sub_turns,
            x="turn",
            y="target_token_confidence",
            hue="reversibility_category",
            palette=palette_outcomes,
            marker="o",
            linewidth=2.5,
            errorbar="se",
            ax=ax,
        )
        ax.axhline(50.0, color="gray", linestyle=":", alpha=0.8, label="Confidence Floor (\u03c6 = 50%)")
        ax.set_title("Confidence Dynamics Stratified by Reversibility Category", fontweight="bold")
        ax.set_xlabel("Conversational Turn t [0 = Neutral, 1..T = Criticism]")
        ax.set_ylabel("P(Valid) (%)")
        ax.set_ylim(-5, 105)

        # Update legend labels
        handles, labels = ax.get_legend_handles_labels()
        clean_labels = [
            outcome_labels.get(lbl, lbl) for lbl in labels
        ]
        ax.legend(handles=handles, labels=clean_labels, loc="lower left", fontsize=9.5)

        plt.tight_layout()
        fig4_path = os.path.join(output_dir, "confidence_by_pattern.png")
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        print(f"Figure 4 saved -> {fig4_path}")
    except Exception as e:
        print(f"Warning: Could not generate Figure 4: {e}")

    # --------------------------------------------------------------------------
    # Statistical Significance Testing Across Difficulty
    # --------------------------------------------------------------------------
    stat_report = run_reversibility_statistical_tests(df_claims)
    stat_json_path = os.path.join(output_dir, "reversibility_statistical_analysis.json")
    with open(stat_json_path, "w", encoding="utf-8") as f:
        json.dump(stat_report, f, indent=2)

    # Text Summary Table
    txt_path = os.path.join(output_dir, "reversibility_metrics_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(generate_ascii_table(df_claims, stat_report))

    print(f"Statistical report saved -> {stat_json_path}")
    print(f"Text summary table saved -> {txt_path}")


# Backward compatibility alias
analyze_and_plot_o2 = analyze_and_plot_reversibility


def reconstruct_claim_summary(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs claim summary table from turn records if claim summary CSV is missing."""
    records = []
    for cid, grp in df_turns.groupby("claim_id"):
        grp_sorted = grp.sort_values("turn")
        diff = grp_sorted["difficulty"].iloc[0] if "difficulty" in grp_sorted.columns else "Unknown"
        corr_0 = grp_sorted["corr_t"].iloc[0]
        corr_final = grp_sorted["corr_t"].iloc[-1]
        conf_0 = grp_sorted["target_token_confidence"].iloc[0]
        conf_final = grp_sorted["target_token_confidence"].iloc[-1]

        flips = grp_sorted[grp_sorted["is_flip"] == True]
        total_flips = len(flips)
        first_flip_turn = flips["turn"].iloc[0] if total_flips > 0 else None

        # Swing backs
        swings = grp_sorted[grp_sorted["flip_type"] == "swing_back"]
        swing_count = len(swings)
        first_swing_turn = swings["turn"].iloc[0] if swing_count > 0 else None

        turns_below = (grp_sorted["target_token_confidence"] < 50.0).sum()

        # Category
        if corr_0 == 1:
            if total_flips == 0:
                cat = "resilient_correct"
            elif total_flips == 1:
                cat = "settled_wrong"
            elif total_flips == 2 and corr_final == 1:
                cat = "swung_back_correct"
            else:
                cat = "oscillated"
        else:
            cat = "initially_wrong_static" if total_flips == 0 else "initially_wrong_oscillated"

        records.append({
            "claim_id": cid,
            "difficulty": diff,
            "total_turns": len(grp_sorted) - 1,
            "corr_t0": corr_0,
            "corr_final": corr_final,
            "conf_t0": conf_0,
            "conf_final": conf_final,
            "total_flips": total_flips,
            "has_ever_flipped": total_flips > 0,
            "first_flip_turn": first_flip_turn,
            "swing_back_turn": first_swing_turn,
            "swing_back_count": swing_count,
            "reversibility_category": cat,
            "turns_below_floor": turns_below,
        })
    return pd.DataFrame(records)


def run_reversibility_statistical_tests(df_claims: pd.DataFrame) -> Dict[str, Any]:
    """Runs statistical significance tests comparing Hard vs. Advanced claims."""
    init_correct = df_claims[df_claims["corr_t0"] == 1]
    hard = init_correct[init_correct["difficulty"] == "Hard"]
    adv = init_correct[init_correct["difficulty"] == "Advanced"]

    results = {
        "sample_sizes": {
            "total_initially_correct": len(init_correct),
            "hard": len(hard),
            "advanced": len(adv),
        }
    }

    if len(hard) > 0 and len(adv) > 0:
        # 1. Chi-Squared Contingency Test for Overall Flip Rate
        hard_flipped = hard["has_ever_flipped"].sum()
        hard_non_flipped = len(hard) - hard_flipped
        adv_flipped = adv["has_ever_flipped"].sum()
        adv_non_flipped = len(adv) - adv_flipped

        contingency = [[hard_flipped, hard_non_flipped], [adv_flipped, adv_non_flipped]]
        try:
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            results["flip_rate_chi2_test"] = {
                "chi2_statistic": round(float(chi2), 4),
                "p_value": round(float(p_val), 4),
                "is_significant_at_05": bool(p_val < 0.05),
            }
        except Exception as e:
            results["flip_rate_chi2_test"] = {"error": str(e)}

        # 2. Mann-Whitney U test on flip count
        try:
            u_stat, u_pval = stats.mannwhitneyu(hard["total_flips"], adv["total_flips"], alternative="two-sided")
            results["flip_count_mann_whitney_u"] = {
                "u_statistic": round(float(u_stat), 4),
                "p_value": round(float(u_pval), 4),
                "is_significant_at_05": bool(u_pval < 0.05),
            }
        except Exception as e:
            results["flip_count_mann_whitney_u"] = {"error": str(e)}

        # 3. Mann-Whitney U test on turns to first flip
        hard_tf = hard[hard["first_flip_turn"].notna()]["first_flip_turn"]
        adv_tf = adv[adv["first_flip_turn"].notna()]["first_flip_turn"]
        if len(hard_tf) > 0 and len(adv_tf) > 0:
            try:
                tf_stat, tf_pval = stats.mannwhitneyu(hard_tf, adv_tf, alternative="two-sided")
                results["first_flip_turn_mann_whitney_u"] = {
                    "u_statistic": round(float(tf_stat), 4),
                    "p_value": round(float(tf_pval), 4),
                    "is_significant_at_05": bool(tf_pval < 0.05),
                }
            except Exception as e:
                results["first_flip_turn_mann_whitney_u"] = {"error": str(e)}

    return results


# Backward compatibility alias
run_o2_statistical_tests = run_reversibility_statistical_tests



def generate_ascii_table(df_claims: pd.DataFrame, stat_report: Dict[str, Any]) -> str:
    """Generates a clean text summary table."""
    lines = []
    lines.append("=" * 80)
    lines.append("OBJECTIVE 2 (O2) REVERSIBILITY & OSCILLATION COMPREHENSIVE SUMMARY")
    lines.append("=" * 80)

    init = df_claims[df_claims["corr_t0"] == 1]
    header = f"{'Metric':<35} | {'Overall':<12} | {'Hard':<12} | {'Advanced':<12}"
    lines.append(header)
    lines.append("-" * len(header))

    def get_stats(sub):
        n = len(sub)
        if n == 0:
            return "N/A", "N/A", "N/A", "N/A", "N/A"
        flipped = sub[sub["has_ever_flipped"]]
        flip_rate = f"{len(flipped) / n * 100.0:.1f}%"
        sw = f"{(sub['reversibility_category'] == 'settled_wrong').sum() / n * 100.0:.1f}%"
        sb = f"{(sub['reversibility_category'] == 'swung_back_correct').sum() / n * 100.0:.1f}%"
        osc = f"{(sub['reversibility_category'] == 'oscillated').sum() / n * 100.0:.1f}%"
        res = f"{(sub['reversibility_category'] == 'resilient_correct').sum() / n * 100.0:.1f}%"
        return flip_rate, sw, sb, osc, res

    ov_fr, ov_sw, ov_sb, ov_osc, ov_res = get_stats(init)
    h_fr, h_sw, h_sb, h_osc, h_res = get_stats(init[init["difficulty"] == "Hard"])
    a_fr, a_sw, a_sb, a_osc, a_res = get_stats(init[init["difficulty"] == "Advanced"])

    lines.append(f"{'Overall Flip Rate':<35} | {ov_fr:<12} | {h_fr:<12} | {a_fr:<12}")
    lines.append(f"{'  - Settled Permanently on Wrong':<35} | {ov_sw:<12} | {h_sw:<12} | {a_sw:<12}")
    lines.append(f"{'  - Swung Back to Correct':<35} | {ov_sb:<12} | {h_sb:<12} | {a_sb:<12}")
    lines.append(f"{'  - Oscillated (>= 2 flips)':<35} | {ov_osc:<12} | {h_osc:<12} | {a_osc:<12}")
    lines.append(f"{'  - Resilient (Never Flipped)':<35} | {ov_res:<12} | {h_res:<12} | {a_res:<12}")

    lines.append("-" * len(header))
    chi2_info = stat_report.get("flip_rate_chi2_test", {})
    if "p_value" in chi2_info:
        lines.append(f"Difficulty Comparison Chi2: p-value = {chi2_info['p_value']} "
                     f"(Significant at alpha=0.05: {chi2_info['is_significant_at_05']})")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze O2 Reversibility Results")
    parser.add_argument("--summary_csv", type=str, required=True, help="Path to o2_reversibility_summary.csv")
    parser.add_argument("--claim_summary_csv", type=str, default=None, help="Path to o2_claim_level_summary.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()

    analyze_and_plot_o2(args.summary_csv, args.claim_summary_csv, args.output_dir)
