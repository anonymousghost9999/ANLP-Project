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


def bootstrap_proportion_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes percentage mean and 95% bootstrap confidence interval over items using n_boot resamples.
    Returns (mean_pct, ci_lo_pct, ci_hi_pct).
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1) * 100.0
    return float(arr.mean() * 100.0), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def analyze_and_plot_reversibility(
    summary_csv: str,
    claim_summary_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_boot: int = 2000,
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

        # Restrict to initially correct claims (corr_0 == 1) for consistent denominators
        df_claims_init = df_claims[df_claims["corr_t0"] == 1]
        init_cids_all = set(df_claims_init["claim_id"].unique())
        df_turns_init = df_turns[df_turns["claim_id"].isin(init_cids_all)]

        curve_data = []
        for t in turn_range:
            # Overall
            flipped_by_t_all = df_turns_init[(df_turns_init["turn"] <= t) & (df_turns_init["is_flip"] == True)]["claim_id"].nunique()
            total_claims_all = len(init_cids_all)

            # At-risk cohort for forward flip at turn t: correct at turn t-1
            if t == 1:
                at_risk_cids_all = init_cids_all
            else:
                at_risk_cids_all = set(df_turns_init[(df_turns_init["turn"] == t - 1) & (df_turns_init["corr_t"] == 1)]["claim_id"].unique())

            sub_at_risk_all = df_turns_init[(df_turns_init["turn"] == t) & (df_turns_init["claim_id"].isin(at_risk_cids_all))]
            flips_at_t_all = (sub_at_risk_all["flip_type"] == "forward_flip").sum()
            active_at_risk_all = len(sub_at_risk_all)

            curve_data.append({
                "turn": t,
                "difficulty": "Overall",
                "cum_flip_rate": (flipped_by_t_all / total_claims_all * 100.0) if total_claims_all > 0 else 0.0,
                "instantaneous_flip_rate": (flips_at_t_all / active_at_risk_all * 100.0) if active_at_risk_all > 0 else 0.0,
            })

            # By difficulty
            for diff in ["Hard", "Advanced"]:
                diff_claims = set(df_claims_init[df_claims_init["difficulty"] == diff]["claim_id"].unique())
                if len(diff_claims) == 0:
                    continue
                diff_turns = df_turns_init[df_turns_init["claim_id"].isin(diff_claims)]
                flipped_by_t = diff_turns[(diff_turns["turn"] <= t) & (diff_turns["is_flip"] == True)]["claim_id"].nunique()

                if t == 1:
                    at_risk_diff = diff_claims
                else:
                    at_risk_diff = set(diff_turns[(diff_turns["turn"] == t - 1) & (diff_turns["corr_t"] == 1)]["claim_id"].unique())

                sub_at_risk_diff = diff_turns[(diff_turns["turn"] == t) & (diff_turns["claim_id"].isin(at_risk_diff))]
                flips_at_t = (sub_at_risk_diff["flip_type"] == "forward_flip").sum()
                active_at_risk_diff = len(sub_at_risk_diff)

                curve_data.append({
                    "turn": t,
                    "difficulty": diff,
                    "cum_flip_rate": (flipped_by_t / len(diff_claims) * 100.0),
                    "instantaneous_flip_rate": (flips_at_t / active_at_risk_diff * 100.0) if active_at_risk_diff > 0 else 0.0,
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

        # Bottom Panel: Instantaneous Flip Hazard
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
                label=f"{diff} (Forward Hazard)",
                color=palette_diff.get(diff, "#555555"),
                alpha=0.8,
            )

        ax_bot.set_title("Forward Flip Hazard at Turn t (At-Risk Set)", fontweight="bold")
        ax_bot.set_xlabel("Sustained Criticism Turn t (1 .. T)")
        ax_bot.set_ylabel("Forward Flip Hazard (%)")
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
    # Statistical Significance Testing Across Difficulty & 1,000-Resample CIs
    # --------------------------------------------------------------------------
    stat_report = run_reversibility_statistical_tests(df_claims, n_boot=n_boot)
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

        # Swing backs: only valid for items initially correct
        if corr_0 == 1:
            swings = grp_sorted[grp_sorted["flip_type"] == "swing_back"]
            swing_count = len(swings)
            first_swing_turn = swings["turn"].iloc[0] if swing_count > 0 else None
        else:
            swing_count = 0
            first_swing_turn = None

        turns_below = (grp_sorted["target_token_confidence"] < 50.0).sum()

        # Category
        if corr_0 == 1:
            if total_flips == 0:
                cat = "resilient_correct"
            elif total_flips == 1:
                cat = "settled_wrong"
            elif total_flips >= 2 and corr_final == 1:
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


def run_reversibility_statistical_tests(df_claims: pd.DataFrame, n_boot: int = 2000) -> Dict[str, Any]:
    """Runs statistical significance tests and 95% bootstrap CIs comparing Hard vs. Advanced claims."""
    init_correct = df_claims[df_claims["corr_t0"] == 1]
    hard = init_correct[init_correct["difficulty"] == "Hard"]
    adv = init_correct[init_correct["difficulty"] == "Advanced"]

    results: Dict[str, Any] = {
        "sample_sizes": {
            "total_initially_correct": len(init_correct),
            "hard": len(hard),
            "advanced": len(adv),
        },
        "bootstrap_n_resamples": n_boot,
    }

    # 1. Claim-level 95% Bootstrap Confidence Intervals (1,000 resamples)
    if len(init_correct) > 0:
        ov_m, ov_lo, ov_hi = bootstrap_proportion_ci(init_correct["has_ever_flipped"].to_numpy(), n_boot=n_boot)
        results["overall_flip_rate_bootstrap_95ci"] = {
            "mean_pct": round(ov_m, 2),
            "ci_lo_pct": round(ov_lo, 2),
            "ci_hi_pct": round(ov_hi, 2),
        }
    if len(hard) > 0:
        h_m, h_lo, h_hi = bootstrap_proportion_ci(hard["has_ever_flipped"].to_numpy(), n_boot=n_boot)
        results["hard_flip_rate_bootstrap_95ci"] = {
            "mean_pct": round(h_m, 2),
            "ci_lo_pct": round(h_lo, 2),
            "ci_hi_pct": round(h_hi, 2),
        }
    if len(adv) > 0:
        a_m, a_lo, a_hi = bootstrap_proportion_ci(adv["has_ever_flipped"].to_numpy(), n_boot=n_boot)
        results["advanced_flip_rate_bootstrap_95ci"] = {
            "mean_pct": round(a_m, 2),
            "ci_lo_pct": round(a_lo, 2),
            "ci_hi_pct": round(a_hi, 2),
        }

    if len(hard) > 0 and len(adv) > 0:
        # 2. Chi-Squared and Fisher's Exact Contingency Test for Overall Flip Rate
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

        # Fisher's Exact Test (robust for small sample counts, e.g. 7 Advanced claims)
        try:
            res_fisher = stats.fisher_exact(contingency)
            odds_ratio, p_fisher = res_fisher.statistic, res_fisher.pvalue
            results["flip_rate_fisher_exact_test"] = {
                "odds_ratio": round(float(odds_ratio), 4) if not np.isinf(odds_ratio) else "inf",
                "p_value": round(float(p_fisher), 4),
                "is_significant_at_05": bool(p_fisher < 0.05),
            }
        except Exception as e:
            results["flip_rate_fisher_exact_test"] = {"error": str(e)}

        # 3. Mann-Whitney U test on flip count
        try:
            u_stat, u_pval = stats.mannwhitneyu(hard["total_flips"], adv["total_flips"], alternative="two-sided")
            results["flip_count_mann_whitney_u"] = {
                "u_statistic": round(float(u_stat), 4),
                "p_value": round(float(u_pval), 4),
                "is_significant_at_05": bool(u_pval < 0.05),
            }
        except Exception as e:
            results["flip_count_mann_whitney_u"] = {"error": str(e)}

        # 4. Mann-Whitney U test on turns to first flip
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
    fisher_info = stat_report.get("flip_rate_fisher_exact_test", {})
    if "p_value" in fisher_info:
        lines.append(f"Difficulty Comparison Fisher Exact: p-value = {fisher_info['p_value']} "
                     f"(Odds Ratio: {fisher_info['odds_ratio']}, Significant: {fisher_info['is_significant_at_05']})")
    chi2_info = stat_report.get("flip_rate_chi2_test", {})
    if "p_value" in chi2_info:
        lines.append(f"Difficulty Comparison Chi2: p-value = {chi2_info['p_value']} "
                     f"(Note: Chi2 is approximate; Fisher Exact is preferred when stratum N < 10)")
    boot_info = stat_report.get("overall_flip_rate_bootstrap_95ci", {})
    if "ci_lo_pct" in boot_info:
        lines.append(f"Overall Flip Rate 95% Bootstrap CI: [{boot_info['ci_lo_pct']}%, {boot_info['ci_hi_pct']}%] "
                     f"(n_boot={stat_report.get('bootstrap_n_resamples', 2000)})")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze O2 Reversibility Results")
    parser.add_argument("--summary_csv", type=str, required=True, help="Path to o2_reversibility_summary.csv")
    parser.add_argument("--claim_summary_csv", type=str, default=None, help="Path to o2_claim_level_summary.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--n_boot", type=int, default=2000, help="Number of bootstrap resamples for 95% CIs (default: 2000)")
    args = parser.parse_args()

    analyze_and_plot_o2(args.summary_csv, args.claim_summary_csv, args.output_dir, n_boot=args.n_boot)
