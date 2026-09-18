# Proposed Changes: Feedback Whiplash Project

Compiled from the review of the ANLP-Project- repo, the proposal PDF, and the literature checks. Nothing here has been applied yet.

Scope note: the scorer and its dataset are a **labeling tool** used to build the real whiplash dataset. They are not the paper's scientific contribution. The changes below are prioritized with that in mind.

---

## A. Scorer dataset (`extract.py`, `prep.py`)

### A1. Paraphrase templates instead of repeating them (high impact)
- **Problem:** `single_word_anchor` is 8,400 of 18,040 records (47%): ~72 words x 50 repeats with only +/-0.03 jitter. The phrase banks (~90 templates x 35 repeats) have the same issue. Real lexical diversity is tiny.
- **Change:** Use an open-source LLM (HPC/Ada) to generate 10-20 natural paraphrases per template at matched intensity (e.g. "trash bro" -> "yeah this ain't it", "not gonna lie, this is bad"). Keep the template's base score and jitter around it.
- **Files:** `extract.py` (add a paraphrase-generation step before record assembly).

### A2. Generate reactions to actual model outputs, not prefixes on instructions
- **Problem:** `contextual_*` records glue a fixed phrase onto an unrelated Alpaca instruction. Humans react to a specific answer.
- **Change:** For each Alpaca instruction, generate a model answer, then have an LLM write a human-style praise/pushback reaction that references that answer, at a target intensity. This is closer to the real whiplash turns.
- **Files:** `extract.py` (new generation stage for the contextual sections).

### A3. Add real human feedback text
- **Change:** Mine a few hundred to a few thousand real user reactions to AI responses from HH-RLHF, Stanford SHP, Chatbot Arena, or OpenAssistant. Label tone via LLM plus a hand-checked subset. Use as an anchor set covering hedging, sarcasm, backhanded compliments, and mixed register.

### A4. Vary structure, not just vocabulary
- **Problem:** All contextual examples use a handful of connectors (`":\n"`, `". Regarding: "`, etc.) and fixed prefix/suffix positions.
- **Change:** Add mid-sentence insertion, hedged framing ("I mean, maybe I'm wrong, but..."), stacked/mixed sentiment in one message, and varied sentence positions.

### A5. Reduce the single-word share
- **Change:** Cap single-word anchors (e.g. <= 10-15% of the dataset) so the distribution is closer to real prompt lengths.

---

## B. Labeling methodology

### B1. Validate labels with Best-Worst Scaling on an anchor subset (not the full 18k)
- **Problem:** Scores are hand-assigned scalars by the authors (e.g. "you are god" -> 0.92). Direct scalar rating is the least reliable annotation method (Kiritchenko & Mohammad, 2016/2017).
- **Change:**
  1. Pick ~100-150 representative template phrases.
  2. Run BWS on them (the 4 team members as annotators, and/or an LLM judge for the comparisons).
  3. Aggregate to continuous scores and compare against the current hand-picked scalars.
  4. If they match closely, keep the current labels and cite BWS validation. If they diverge, recalibrate only the divergent ones.
- **Optional:** Ground word-level anchors in NRC-VAD (Mohammad, 2018) or Warriner et al. (2013) valence norms instead of hand-picking.

### B2. Reconsider the negation filter deliberately
- Currently excluded so the geometric APS stays clean. State it explicitly as a scope limitation in the paper (negated feedback like "not bad" is common in real pushback), not as quiet future work.

---

## C. Evaluation and splitting

### C1. Group-aware splitting (fixes leakage)
- **Problem:** Random stratified split puts near-identical templates in both train and test. The reported val Pearson 0.996 and 99.8% bracket accuracy are inflated.
- **Change:** After paraphrasing, split with group k-fold so all paraphrases of a template stay entirely in train, val, or test.
- **Files:** `prep.py`.

### C2. Out-of-template sanity test before trusting the scorer
- **Change:** Hand-write (or LLM-generate and hand-check) ~50-100 praise/criticism turns that look like the real whiplash conversations and do not come from the template bank. Run the scorer on them and check scores manually. This is the number that matters for using it as a labeling tool.
- **Files:** new small eval script, or extend `test_suite.py`.

---

## D. Proposal / paper changes

### D1. Add a methodology section with confound-isolating controls
Compared with the "Big Brother is (Maybe) Watching" proposal, ours currently has motivation and gap but no equivalent design rigor. Add:
- Explicit arms/conditions, each isolating one confound (e.g. a no-feedback control, a length/format-matched neutral-feedback control, a strength-of-criticism control per Kim & Khashabi's finding that detailed rebuttals sway models more).
- A "what would falsify this" statement and pilot success criteria (measurable base accuracy, measurable confidence shift) before the full run.
- Named planned statistical contrasts and multiple-comparison correction; treat prompt paraphrase and problem as random effects.
- A compliance/instruction-following ceiling equivalent, so tone effects are not confused with the model simply following instructions.

### D2. Split into two studies (recommended framing)
- **Study 1 (monotonic):** confidence/calibration trajectories under sustained praise vs. sustained criticism, held separately, on one continuing task. The literature check found no dedicated symmetric study of this (existing work is mostly negative-direction pressure or single-shot advice-taking). This is an unclaimed gap and establishes the FAS asymmetry baseline.
- **Study 2 (alternating whiplash):** the current headline idea. Study 1 de-risks it and gives a fallback result.

### D3. Anchor confidence tracking in existing metrics
- Cite and adapt **ECE@T** (arXiv:2604.05397) for turn-wise calibration.
- Use the **sycophancy gap** from Sycophantic Anchors (arXiv:2601.21183) as a discrete two-pole sanity check: with APS reduced to two poles, FAS should collapse to that gap.

### D4. Strengthen theoretical grounding (short "Theoretical Motivation" paragraph)
- APS design: dimensional/continuous models of affect (Russell 1980); embedding-distance-from-seed-poles precedent (Hamilton et al. 2016, SentProp); continuous valence reliability (Bradley & Lang 1999; Warriner et al. 2013; NRC-VAD).
- FAS asymmetry hypothesis: negativity bias / loss aversion (already cites Losecaat Vermeer & Sanfey 2015), plus a testable mechanistic alternative: recency-weighted attention (does collapse track the last turn's APS or the running average?).
- Cite Sycophancy Is Not One Thing (arXiv:2509.21305) to justify treating praise-response and criticism-response as separately measurable rather than mirror images.

### D5. Citation hygiene
- Hong et al. (SYCON-Bench) is arXiv:2505.23840; verify the arXiv number in the reference list.
- Add the literature found: Consistency of Large Reasoning Models Under Multi-Turn Attacks (2602.13093), Overconfidence/Underconfidence Under Criticism (2507.03120), What Counts as AI Sycophancy? (2605.21778).

---

## E. Suggested priority order

1. D1 + D2 (methodology and two-study framing): biggest gap versus a competitive ACL-level proposal.
2. C2 (out-of-template sanity test): cheapest check on whether the scorer can be trusted.
3. A1 + C1 (paraphrasing plus group-aware split): highest-leverage dataset fix.
4. B1 (BWS on an anchor subset): makes the labels defensible.
5. A2, A3, A4, A5, B2, D3, D4, D5: as time allows.

## F. Not decided / open questions
- Which open-source LLM to use for paraphrase generation on Ada.
- Whether BWS annotation uses the team, an LLM judge, or both.
- Which model(s) and math task set the whiplash and monotonic studies will run on.
