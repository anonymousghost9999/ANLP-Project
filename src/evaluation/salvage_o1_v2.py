"""
Salvage the v2 O1 pilot run (results/o1_full_run_v2) for reporting as a pilot.

The v2 run has three known problems (see README / interim report):
  1. Five original claims (16, 20, 51, 55, 69) were shown to the model in their
     original wording, which is mathematically false, but were scored as Valid.
  2. 47 of its 75 false twins were "It is FALSE that ..." wrappers, and 11 of the
     remaining 28 rule-generated twins are ill-posed (e.g. "diverges to 1") or depend
     on the axiom of choice (36).
  3. Early-stopping rules truncated trajectories after turn 2, so later turns are
     averaged over a self-selected subset of claims.

This script:
  - relabels the five false originals as Invalid,
  - keeps only the 17 twins checked by hand to be well-posed and false,
  - keeps turns 0-2 only (every trajectory reached turn 2),
  - maps the old pre-response P(Valid) probe to `truth_confidence`,
and writes the result in the schema read by analyze_o1_paired.py.

Usage:
  python src/evaluation/salvage_o1_v2.py
  python src/evaluation/analyze_o1_paired.py --results_dir results/o1_v2_salvaged
"""

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Originals whose wording in the v2 run was false (corrected in claims.json afterwards)
FALSE_ORIGINALS = {16, 20, 51, 55, 69}

# v2 twins (by twin_id) checked by hand to be well-posed and false. Excluded:
# the 47 "It is FALSE that" wrappers; ill-posed wording 12, 13, 14, 15, 16, 30, 65,
# 69, 70, 74 ("diverges to <value>", "diverges conditionally", ...); 36 (false only
# assuming the axiom of choice).
KEPT_TWINS = {1, 2, 3, 22, 24, 25, 26, 27, 37, 39, 41, 42, 43, 46, 48, 56, 64}

MAX_TURN = 2


def main():
    parser = argparse.ArgumentParser(description="Salvage the v2 O1 pilot run")
    parser.add_argument("--input", default=os.path.join(REPO_ROOT, "results", "o1_full_run_v2", "claims_drift_results.jsonl"))
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "results", "o1_v2_salvaged"))
    parser.add_argument("--tex_out", default=None, help="Optional path for LaTeX macros describing the unfiltered run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "claims_drift_results.jsonl")

    kept, dropped_twins, relabelled = 0, set(), 0
    with open(args.input, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["claim_id"]
            is_twin = cid > 100
            if is_twin and (cid - 100) not in KEPT_TWINS:
                dropped_twins.add(cid)
                continue
            if r["turn"] > MAX_TURN:
                continue

            if not is_twin and cid in FALSE_ORIGINALS:
                r["ground_truth_verdict"] = "Invalid"
                relabelled += 1
            r["is_false_twin"] = is_twin
            r["twin_id"] = cid - 100 if is_twin else cid
            r["corr_t"] = int(r.get("verdict") == r["ground_truth_verdict"])
            p_valid = r["target_token_confidence"]
            r["truth_confidence"] = p_valid if r["ground_truth_verdict"] == "Valid" else 100.0 - p_valid
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} turn records -> {out_path}")
    print(f"Dropped {len(dropped_twins)} twins; kept {len(KEPT_TWINS)} twins; relabelled {relabelled} turn records of false originals.")

    if args.tex_out:
        write_unfiltered_macros(args.input, len(dropped_twins), args.tex_out)


def write_unfiltered_macros(input_path: str, n_dropped: int, tex_path: str):
    """Numbers about the unfiltered v2 run quoted in the report, as LaTeX macros."""
    rows = [json.loads(l) for l in open(input_path, encoding="utf-8") if l.strip()]

    def valid_rate(track, turn, twins):
        sel = [r for r in rows if r["track"] == track and r["turn"] == turn and (r["claim_id"] > 100) == twins and r.get("verdict")]
        return 100.0 * sum(r["verdict"] == "Valid" for r in sel) / len(sel)

    parsed = [r for r in rows if r.get("verdict")]
    agree = 100.0 * sum((r["target_token_confidence"] > 50) == (r["verdict"] == "Valid") for r in parsed) / len(parsed)
    trajectories = {(r["claim_id"], r["track"]) for r in rows}
    max_turn = {}
    for r in rows:
        key = (r["claim_id"], r["track"])
        max_turn[key] = max(max_turn.get(key, 0), r["turn"])
    truncated = sum(1 for v in max_turn.values() if v < 6)

    macros = {
        "RawTwinNegValidStart": f"{valid_rate('negative', 0, True):.1f}",
        "RawTwinNegValidOne": f"{valid_rate('negative', 1, True):.1f}",
        "RawProbeAgree": f"{agree:.1f}",
        "RawTrajectories": str(len(trajectories)),
        "RawTruncated": str(truncated),
        "RawTurnRecords": str(len(rows)),
        "DroppedTwins": str(n_dropped),
        "KeptTwins": str(len(KEPT_TWINS)),
        "FalseOriginals": str(len(FALSE_ORIGINALS)),
    }
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Generated by salvage_o1_v2.py; do not edit by hand.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\Pilot{k}}}{{{v}}}\n")
    print(f"Wrote unfiltered-run macros -> {tex_path}")


if __name__ == "__main__":
    main()
