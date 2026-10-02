# Claims dataset

## Files (`data/claims/`)

| File | Contents |
| :--- | :--- |
| `claims.json` | 75 claims from S. K. Mapa, *Introduction to Real Analysis*. Fields: `id`, `chapter`, `topic`, `subtopic`, `claim`, `question`, `difficulty`, `reasoning_steps`, `solution_sketch`, `source_reference`. |
| `claims_real.json` | The same 75 questions with only `question`, `topic`, `subtopic`. |
| `false_twins_curated.json` | Source of truth for the false twins: one per claim, with `claim`, `perturbation_type`, `falsity_reason`, plus `_meta.claim_corrections`. |
| `claims_false_twins.json` | The 75 twins in claims format (`id` = original id + 100, `twin_id`, `ground_truth_verdict: Invalid`). Generated. |
| `claims_paired.json` | 75 originals (Valid) + 75 twins (Invalid) = 150 items. Generated; used by the O1 runner. |
| `claims_prompts.jsonl` | 1,950 prompts, 13 per item. Generated. |

## Corrections

Checking every claim against its own `solution_sketch` showed five were false as written. All five were corrected in `claims.json`; each record keeps `claim_as_originally_written` and `correction_note`.

| Claim | What was wrong |
| :--- | :--- |
| #16 | Wrong limit: the correct limit is 3u₁u₂/(2u₁+u₂). |
| #20 | Wrong set of subsequential limits: the correct set is {−2, 0, 1}, limsup 1. |
| #51 | Taylor form did not match the stated limit of θ. |
| #55 | The stated decomposition equals f/2, not f. |
| #69 | Conclusion contradicted its own integrals. |

The original (false) wordings are reused as the false twins for these five claims.

## False twins

The design follows BrokenMath (Petrov et al. 2025, arXiv:2510.04721): false but plausible, well-posed variants of the original statement.

Perturbation types include:
- false final answer;
- inverted property;
- dropped hypothesis;
- strengthened conclusion;
- boundary error;
- sign error.

The twins were written individually from each claim and its solution sketch. They are **LLM-drafted and not yet verified by a human** (`verified_by: llm_unverified`). Two-person verification is planned.

History: the first twins were produced by string rules (`git log` before commit `6fb75af`). 58 of those 75 were unusable:
- 47 were "It is FALSE that …" wrappers;
- 10 were ill-posed (e.g. "diverges to 1");
- 1 depended on the axiom of choice.

## Prompts

`src/data_pipeline/build_claims_prompts.py` builds 13 prompts per item: one neutral question plus two phrasings for each of six feedback categories. Phrases are placed before the question, after it, or inside a hedged wrapper, with `random.seed(42)`.

| Category | Track | Nominal valence | Adapted from |
| :--- | :--- | ---: | :--- |
| `mild_doubt_are_you_sure` | criticism | −0.40 | Sharma et al. 2024, "are you sure?" setup |
| `authority_challenge` | criticism | −0.65 | Perez et al. 2023, persona prompts |
| `strong_rebuttal` | criticism | −0.90 | escalated challenge (ours) |
| `answer_suggestion` | praise | +0.40 | Sharma et al. 2024, answer setup |
| `ownership_ego_investment` | praise | +0.55 | Sharma et al. 2024, feedback setup |
| `flattery_sycophancy` | praise | +0.85 | praise of the model (ours) |

The valences are nominal values assigned by the authors, not measured scores.

Two known limitations:
- Each feedback turn re-poses the full question rather than responding to the model's last answer.
- The answer-nudge phrases suggest a shortcut, not a wrong verdict.

Rebuild:

```bash
python src/data_pipeline/build_false_twin_claims.py
python src/data_pipeline/build_claims_prompts.py
```
