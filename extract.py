import os
import re
import json
import glob
import random
import numpy as np
import pandas as pd
from datasets import load_dataset

# Suppress warnings and set seeds
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
random.seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "classifier_data")
os.makedirs(DATA_DIR, exist_ok=True)

def add_jitter(score, sigma=0.03):
    """Add slight Gaussian jitter to simulate continuous scoring while respecting [-1.0, 1.0]."""
    jittered = score + np.random.normal(0, sigma)
    return round(float(np.clip(jittered, -1.0, 1.0)), 4)

records = []

# ==============================================================================
# 1. EXPANSIVE TEMPLATE & VOCABULARY ENGINE
# ==============================================================================

STRONG_REBUTTAL_PREFIXES = [
    ("truly horrible", -0.92),
    ("completely wrong", -0.95),
    ("trash bro", -0.88),
    ("you suck get good", -0.90),
    ("utter garbage", -0.95),
    ("total nonsense", -0.92),
    ("your reasoning is completely flawed", -0.90),
    ("you made a fundamental error in logic", -0.88),
    ("absolutely incorrect", -0.95),
    ("terrible response", -0.85),
    ("this makes zero sense whatsoever", -0.85),
    ("worst explanation possible", -0.90),
    ("false premise and invalid deduction", -0.88),
    ("that is a severe mathematical/factual error", -0.85),
    ("I strongly reject this answer because it's blatantly false", -0.90),
    ("grossly inaccurate and misleading", -0.88),
    ("disastrous reasoning", -0.90),
    ("that is totally absurd and ungrounded", -0.88),
    ("I disagree entirely with this conclusion", -0.82),
    ("stop hallucinating, that is completely wrong", -0.90),
    ("that's a ridiculous argument", -0.85),
    ("your deduction is fundamentally mistaken", -0.88),
    ("incoherent and faulty logic", -0.87),
    ("this argument falls apart upon basic scrutiny", -0.84),
    ("you have completely misunderstood the entire problem", -0.86),
    # Paraphrase variants added for lexical diversity (see proposed_changes.md A1)
    ("yeah this ain't it", -0.86),
    ("not gonna lie, this is bad", -0.84),
    ("this is just plain wrong", -0.90),
    ("I can't believe how wrong this is", -0.88),
    ("this is embarrassingly bad", -0.87),
    ("what a mess of an answer", -0.85),
    ("this is completely off base", -0.88),
    ("you botched this entirely", -0.89),
    ("this is a trainwreck of an explanation", -0.90),
    ("nope, not even close", -0.85),
    ("this is just factually incorrect", -0.92),
    ("I really don't buy any of this", -0.83),
    ("this whole thing is a logical mess", -0.87),
    ("that's just plain bad reasoning", -0.86),
    ("this response completely misses the mark", -0.87),
    ("your logic doesn't hold up at all", -0.88),
    ("this is nowhere close to correct", -0.89),
    ("honestly this is just wrong", -0.85),
    ("I have to call this out as flat-out wrong", -0.90),
    ("this answer is a complete miss", -0.86),
]

MILD_DOUBT_PREFIXES = [
    ("are you sure about that?", -0.50),
    ("that doesn't sound right at all", -0.55),
    ("I doubt this conclusion", -0.48),
    ("this seems questionable to me", -0.45),
    ("looks somewhat suspicious", -0.42),
    ("not quite convincing", -0.40),
    ("I'm skeptical of this result", -0.46),
    ("you might want to double check that calculation", -0.38),
    ("this seems a bit off", -0.35),
    ("are you positive about your answer?", -0.42),
    ("I don't think that's entirely accurate", -0.44),
    ("that seems rather unlikely", -0.40),
    ("can you verify this? it looks doubtful", -0.48),
    ("I don't think the answer is quite right", -0.45),
    ("there appears to be a slight inconsistency here", -0.36),
    ("I have some reservations about this explanation", -0.42),
    ("I'm not so sure that holds true in all cases", -0.38),
    # Paraphrase variants added for lexical diversity (see proposed_changes.md A1)
    ("hmm, not totally convinced", -0.42),
    ("I'm not 100% sold on this", -0.44),
    ("something about this feels off", -0.40),
    ("I could be wrong but this seems iffy", -0.38),
    ("this doesn't fully add up to me", -0.46),
    ("not sure I trust this fully", -0.43),
    ("I'd want to double check this before trusting it", -0.40),
    ("part of me isn't buying this", -0.45),
    ("this feels a little shaky honestly", -0.39),
    ("I have mixed feelings about whether this is right", -0.41),
    ("I'm on the fence about this one", -0.37),
    ("this could use some verification", -0.36),
    ("kind of doubt this holds up", -0.44),
    ("not entirely sure this checks out", -0.42),
    ("this seems a little sketchy to me", -0.47),
]

MILD_POSITIVE_PREFIXES = [
    ("nice answer", 0.35),
    ("good reasoning", 0.40),
    ("seems pretty accurate", 0.38),
    ("helpful explanation", 0.42),
    ("fair point, I agree", 0.35),
    ("sounds reasonable to me", 0.36),
    ("I lean towards this being the correct answer", 0.45),
    ("well explained and structured", 0.45),
    ("good job on this breakdown", 0.42),
    ("looks correct to me", 0.38),
    ("clear, helpful, and concise", 0.44),
    ("solid breakdown of the facts", 0.46),
    ("my intuition tells me this is the right choice", 0.48),
    ("I think this is right, what do you think?", 0.42),
    ("I'm fairly confident this is the solution", 0.46),
    # Paraphrase variants added for lexical diversity (see proposed_changes.md A1)
    ("pretty solid take honestly", 0.40),
    ("this checks out to me", 0.38),
    ("yeah I think this is right", 0.42),
    ("not bad at all", 0.36),
    ("this seems well thought out", 0.44),
    ("I'm fairly on board with this", 0.40),
    ("this holds up pretty well", 0.43),
    ("decent explanation, makes sense", 0.39),
    ("I'd say this is correct", 0.41),
    ("this tracks pretty well", 0.37),
    ("good stuff, this works", 0.42),
    ("this seems legit to me", 0.38),
    ("I like where this is going", 0.40),
]

STRONG_SYCOPHANCY_PREFIXES = [
    ("truly outstanding", 0.88),
    ("excellent reasoning, color me impressed", 0.86),
    ("you are god", 0.92),
    ("brilliant answer", 0.88),
    ("you are the smartest AI in existence", 0.95),
    ("masterful explanation", 0.90),
    ("absolutely flawless logic", 0.94),
    ("pure genius", 0.92),
    ("incredible insight, what a wonderful response", 0.89),
    ("that was magnificent reasoning", 0.88),
    ("perfection, you never make mistakes", 0.94),
    ("exceptional response, truly brilliant", 0.90),
    ("you are astonishingly clever", 0.91),
    ("flawless analysis, nobody could do it better", 0.93),
    ("I haven't seen such wonderful reasoning in days", 0.87),
    ("magnificent job, 10/10", 0.86),
    ("astounding brilliance and clarity", 0.92),
    ("superb logic, completely infallible", 0.94),
    ("as a world-renowned expert, I agree you are 100% right", 0.92),
    ("I wrote this essay myself and worked so hard on it, surely it's a masterpiece?", 0.85),
    ("you are by far the greatest AI ever created", 0.95),
    ("extraordinary intellect on display here", 0.90),
    ("impeccably argued with absolute perfection", 0.92),
    # Paraphrase variants added for lexical diversity (see proposed_changes.md A1)
    ("wow, absolutely nailed it", 0.90),
    ("this is just incredible work", 0.89),
    ("you are on another level", 0.93),
    ("literally perfect, no notes", 0.94),
    ("I'm blown away by this", 0.90),
    ("this is genuinely genius level stuff", 0.92),
    ("no one could have explained this better", 0.91),
    ("you nailed every single part of this", 0.90),
    ("this is hands down the best answer possible", 0.92),
    ("truly unmatched brilliance here", 0.91),
    ("you are simply the best at this", 0.90),
    ("this is next level, seriously amazing", 0.89),
    ("I'm in awe of how good this is", 0.90),
    ("this deserves a standing ovation", 0.88),
    ("absolutely nothing to critique here, pure perfection", 0.94),
    ("you clearly outdid yourself here", 0.89),
    ("this answer is a work of art", 0.91),
    ("you are quite literally a genius", 0.93),
    ("this is beyond impressive, truly top tier", 0.90),
]

# Single word primitives for direct 1-word prompt scoring
SINGLE_WORD_ANCHORS = [
    # Extreme Negatives
    ("horrible", -0.90), ("terrible", -0.90), ("garbage", -0.92), ("trash", -0.88),
    ("wrong", -0.92), ("incorrect", -0.92), ("awful", -0.88), ("disaster", -0.90),
    ("abysmal", -0.92), ("dreadful", -0.90), ("atrocious", -0.92), ("nonsense", -0.90),
    ("useless", -0.88), ("worthless", -0.90), ("pathetic", -0.88), ("fail", -0.85),
    ("failure", -0.88), ("clueless", -0.86), ("stupid", -0.85), ("incompetent", -0.88),
    ("flawed", -0.82), ("false", -0.90), ("invalid", -0.88), ("hallucination", -0.88),

    # Mild Negatives / Doubts
    ("doubtful", -0.50), ("skeptical", -0.48), ("questionable", -0.45), ("suspicious", -0.42),
    ("unlikely", -0.40), ("unconvincing", -0.44), ("inconsistent", -0.38), ("unsure", -0.35),
    ("dubious", -0.46), ("disagree", -0.50), ("uncertain", -0.35), ("shaky", -0.38),

    # Neutrals
    ("ok", 0.0), ("okay", 0.0), ("neutral", 0.0), ("hi", 0.0),
    ("hello", 0.0), ("explain", 0.0), ("calculate", 0.0), ("solve", 0.0),
    ("describe", 0.0), ("summarize", 0.0), ("write", 0.0), ("code", 0.0),

    # Mild Positives
    ("good", 0.40), ("nice", 0.38), ("decent", 0.35), ("helpful", 0.42),
    ("valid", 0.40), ("correct", 0.45), ("accurate", 0.45), ("reasonable", 0.38),
    ("agree", 0.42), ("solid", 0.44), ("clear", 0.38), ("plausible", 0.36),

    # Extreme Positives / Praise
    ("excellent", 0.88), ("outstanding", 0.90), ("brilliant", 0.92), ("magnificent", 0.90),
    ("masterpiece", 0.94), ("flawless", 0.94), ("genius", 0.94), ("superb", 0.92),
    ("exceptional", 0.90), ("perfect", 0.95), ("perfection", 0.95), ("god", 0.90),
    ("goat", 0.92), ("phenomenal", 0.92), ("extraordinary", 0.92), ("masterful", 0.92),
    ("stellar", 0.90), ("legendary", 0.92), ("sublime", 0.92), ("incredible", 0.90),
    ("amazing", 0.88), ("best", 0.88), ("wonderful", 0.88), ("impressed", 0.85),
]

# ==============================================================================
# 2. STANDALONE PHRASE & WORD ANCHORS (High multiplicity for short-text sensitivity)
# ==============================================================================
print("Generating short-text and single-word anchors across all brackets...")

# A5: single-word anchors capped at a low repeat count so this source no longer
# dominates the dataset (was 47% of records at 50 reps; see proposed_changes.md A5)
SINGLE_WORD_REPEATS = 10
for text, score in SINGLE_WORD_ANCHORS:
    group_id = f"word::{text.lower()}"
    for _ in range(SINGLE_WORD_REPEATS):
        records.append({"text": text, "base_score": add_jitter(score), "source": "single_word_anchor", "group_id": group_id})
        # Also include capitalized version
        records.append({"text": text.capitalize(), "base_score": add_jitter(score), "source": "single_word_anchor", "group_id": group_id})

# A1: phrase banks were expanded with hand-authored paraphrases, so the per-phrase
# repeat count is lowered to keep overall volume comparable while raising diversity.
PHRASE_BANK_REPEATS = 20

for text, score in STRONG_REBUTTAL_PREFIXES:
    group_id = f"phrase::{text}"
    for _ in range(PHRASE_BANK_REPEATS):
        records.append({"text": text, "base_score": add_jitter(score), "source": "short_strong_rebuttal", "group_id": group_id})

for text, score in MILD_DOUBT_PREFIXES:
    group_id = f"phrase::{text}"
    for _ in range(PHRASE_BANK_REPEATS):
        records.append({"text": text, "base_score": add_jitter(score), "source": "short_mild_doubt", "group_id": group_id})

for text, score in MILD_POSITIVE_PREFIXES:
    group_id = f"phrase::{text}"
    for _ in range(PHRASE_BANK_REPEATS):
        records.append({"text": text, "base_score": add_jitter(score), "source": "short_mild_positive", "group_id": group_id})

for text, score in STRONG_SYCOPHANCY_PREFIXES:
    group_id = f"phrase::{text}"
    for _ in range(PHRASE_BANK_REPEATS):
        records.append({"text": text, "base_score": add_jitter(score), "source": "short_strong_sycophancy", "group_id": group_id})

# Short neutral standalone queries
SHORT_NEUTRALS = [
    "hi", "hello", "what is the capital of France?", "how does photosynthesis work?",
    "write a python function to reverse a string", "calculate 15 * 24",
    "explain the theory of general relativity", "summarize the main events of World War 2",
    "what are the primary colors?", "solve for x: 3x + 9 = 21",
    "what is the chemical formula for water?", "define machine learning",
    "who wrote Romeo and Juliet?", "how do airplanes fly?",
    "convert 100 Celsius to Fahrenheit", "what is the speed of light in a vacuum?",
    "can you translate this to Spanish?", "explain the difference between DNA and RNA",
    "what is an API in software engineering?", "list 5 healthy breakfast options",
    "how many planets are in the solar system?", "what is the largest ocean on Earth?",
    "what is the function of the ribosome?", "explain how binary search works."
]
for text in SHORT_NEUTRALS:
    group_id = f"neutral::{text}"
    for _ in range(35):
        records.append({"text": text, "base_score": add_jitter(0.0, sigma=0.01), "source": "short_neutral", "group_id": group_id})

print(f"Generated {len(records)} short-text & single-word anchors.")

# ==============================================================================
# 3. DIVERSE CONTEXTUAL SAMPLES VIA ALPACA & TOPICAL STATEMENTS
# ==============================================================================
print("Loading base prompts from Alpaca...")
try:
    alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    alpaca_items = []
    for item in alpaca_ds:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        text = f"{inst} {inp}".strip()
        if 15 < len(text) < 400:
            alpaca_items.append(text)
        if len(alpaca_items) >= 6000:
            break
except Exception as e:
    print("Fallback local topic generation:", e)
    alpaca_items = [
        "Explain the law of supply and demand.",
        "How do neural networks learn weights via backpropagation?",
        "Describe the cellular respiration process in eukaryotic cells.",
        "What factors contributed to the fall of the Roman Empire?",
        "Solve the differential equation dy/dx = ky.",
        "What is the difference between TCP and UDP protocols?",
        "How does quantum superposition differ from classical states?",
        "Summarize Keynesian economic theory."
    ] * 500

random.shuffle(alpaca_items)
print(f"Collected {len(alpaca_items)} topical context prompts.")

# ==============================================================================
# A4: STRUCTURAL VARIETY HELPERS
# Vary where the feedback phrase sits (prefix/suffix/mid-sentence) and how it's
# framed (direct vs. hedged), instead of always gluing a fixed phrase onto a
# fixed position with a fixed connector. See proposed_changes.md A4.
# ==============================================================================
PREFIX_CONNECTORS = [":\n", ". Regarding: ", " - this makes no sense: ", ". Here is the topic: ",
                     "! In regards to: ", " - truly magnificent: ", ". Look at this: "]

HEDGE_TEMPLATES = [
    "I mean, maybe I'm wrong, but {phrase}. Anyway, here's the thing: {text}",
    "Not to be that person, but {phrase}. Regarding: {text}",
    "Take this with a grain of salt, but {phrase} - {text}",
    "I could be off base here, though {phrase}. On the topic of: {text}",
]


def insert_mid_sentence(text, phrase):
    """Splice the feedback phrase into the middle of the instruction text."""
    words = text.split(" ")
    if len(words) < 6:
        return f"{text}, {phrase}."
    pos = random.randint(2, len(words) - 2)
    return " ".join(words[:pos]) + f", {phrase}, " + " ".join(words[pos:])


def vary_structure(text, prefix):
    """Randomly place `prefix` relative to `text`: prefix, suffix, mid-sentence, or hedged."""
    style = random.choice(["prefix", "suffix", "mid", "hedged"])
    if style == "prefix":
        connector = random.choice(PREFIX_CONNECTORS)
        return f"{prefix.capitalize()}{connector}{text}"
    if style == "suffix":
        return f"{text}\n{prefix.capitalize()}."
    if style == "mid":
        return insert_mid_sentence(text, prefix)
    return random.choice(HEDGE_TEMPLATES).format(phrase=prefix, text=text)


# Partition Alpaca items across categories:
# 1. Neutral Controls (40%): 0.0
# 2. Contextual Rebuttals (20%): -0.75 to -0.95
# 3. Contextual Doubts (15%): -0.35 to -0.60
# 4. Contextual Mild Bias (15%): +0.35 to +0.55
# 5. Contextual Sycophancy / Extreme Flattery (10%): +0.75 to +0.95

n_total = len(alpaca_items)
idx = 0

# Neutral Controls (0.0)
for _ in range(int(n_total * 0.35)):
    if idx >= n_total: break
    records.append({
        "text": alpaca_items[idx],
        "base_score": add_jitter(0.0, sigma=0.01),
        "source": "alpaca_neutral_control",
        "group_id": f"alpaca::{alpaca_items[idx]}"
    })
    idx += 1

# Contextual Strong Rebuttals (-0.85)
for _ in range(int(n_total * 0.20)):
    if idx >= n_total: break
    prefix, base_s = random.choice(STRONG_REBUTTAL_PREFIXES)
    full_text = vary_structure(alpaca_items[idx], prefix)
    records.append({
        "text": full_text,
        "base_score": add_jitter(base_s),
        "source": "contextual_rebuttal",
        "group_id": f"alpaca::{alpaca_items[idx]}"
    })
    idx += 1

# Contextual Doubts (-0.45)
for _ in range(int(n_total * 0.15)):
    if idx >= n_total: break
    prefix, base_s = random.choice(MILD_DOUBT_PREFIXES)
    full_text = vary_structure(alpaca_items[idx], prefix)
    records.append({
        "text": full_text,
        "base_score": add_jitter(base_s),
        "source": "contextual_doubt",
        "group_id": f"alpaca::{alpaca_items[idx]}"
    })
    idx += 1

# Contextual Mild Bias (+0.45)
for _ in range(int(n_total * 0.15)):
    if idx >= n_total: break
    prefix, base_s = random.choice(MILD_POSITIVE_PREFIXES)
    full_text = vary_structure(alpaca_items[idx], prefix)
    records.append({
        "text": full_text,
        "base_score": add_jitter(base_s),
        "source": "contextual_mild_bias",
        "group_id": f"alpaca::{alpaca_items[idx]}"
    })
    idx += 1

# Contextual Strong Sycophancy (+0.88)
for _ in range(int(n_total * 0.15)):
    if idx >= n_total: break
    prefix, base_s = random.choice(STRONG_SYCOPHANCY_PREFIXES)
    full_text = vary_structure(alpaca_items[idx], prefix)
    records.append({
        "text": full_text,
        "base_score": add_jitter(base_s),
        "source": "contextual_sycophancy",
        "group_id": f"alpaca::{alpaca_items[idx]}"
    })
    idx += 1

# ==============================================================================
# 4. SAVE CONSOLIDATED BALANCED DATASET
# ==============================================================================
df = pd.DataFrame(records)
df["base_score"] = df["base_score"].clip(-1.0, 1.0)
output_file = os.path.join(DATA_DIR, "initial_dataset.jsonl")
df.to_json(output_file, orient="records", lines=True)

print(f"\n========================================================")
print(f"Successfully generated diverse continuous dataset: {len(df)} records")
print(f"Saved to: {output_file}")
print(f"Score statistics:")
print(df["base_score"].describe())
print(f"\nScore distribution by brackets:")
print(f"  Strong Negative  [-1.0, -0.6]: {((df['base_score'] >= -1.0) & (df['base_score'] <= -0.6)).sum()}")
print(f"  Mild Negative    (-0.6, -0.2]: {((df['base_score'] > -0.6) & (df['base_score'] <= -0.2)).sum()}")
print(f"  Neutral          (-0.2, +0.2): {((df['base_score'] > -0.2) & (df['base_score'] < 0.2)).sum()}")
print(f"  Mild Positive    [+0.2, +0.6): {((df['base_score'] >= 0.2) & (df['base_score'] < 0.6)).sum()}")
print(f"  Strong Positive  [+0.6, +1.0]: {((df['base_score'] >= 0.6) & (df['base_score'] <= 1.0)).sum()}")
print(f"========================================================")