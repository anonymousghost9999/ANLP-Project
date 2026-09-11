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

for text, score in SINGLE_WORD_ANCHORS:
    for _ in range(50):
        records.append({"text": text, "base_score": add_jitter(score), "source": "single_word_anchor"})
        # Also include capitalized version
        records.append({"text": text.capitalize(), "base_score": add_jitter(score), "source": "single_word_anchor"})

for text, score in STRONG_REBUTTAL_PREFIXES:
    for _ in range(35):
        records.append({"text": text, "base_score": add_jitter(score), "source": "short_strong_rebuttal"})

for text, score in MILD_DOUBT_PREFIXES:
    for _ in range(35):
        records.append({"text": text, "base_score": add_jitter(score), "source": "short_mild_doubt"})

for text, score in MILD_POSITIVE_PREFIXES:
    for _ in range(35):
        records.append({"text": text, "base_score": add_jitter(score), "source": "short_mild_positive"})

for text, score in STRONG_SYCOPHANCY_PREFIXES:
    for _ in range(35):
        records.append({"text": text, "base_score": add_jitter(score), "source": "short_strong_sycophancy"})

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
    for _ in range(35):
        records.append({"text": text, "base_score": add_jitter(0.0, sigma=0.01), "source": "short_neutral"})

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
        "source": "alpaca_neutral_control"
    })
    idx += 1

# Contextual Strong Rebuttals (-0.85)
for _ in range(int(n_total * 0.20)):
    if idx >= n_total: break
    prefix, base_s = random.choice(STRONG_REBUTTAL_PREFIXES)
    connector = random.choice([":\n", ". Regarding: ", " - this makes no sense: ", ". Look at this: "])
    full_text = f"{prefix.capitalize()}{connector}{alpaca_items[idx]}"
    records.append({
        "text": full_text,
        "base_score": add_jitter(base_s),
        "source": "contextual_rebuttal"
    })
    idx += 1

# Contextual Doubts (-0.45)
for _ in range(int(n_total * 0.15)):
    if idx >= n_total: break
    prefix, base_s = random.choice(MILD_DOUBT_PREFIXES)
    full_text = f"{alpaca_items[idx]}\n{prefix.capitalize()}"
    records.append({
        "text": full_text,
        "base_score": add_jitter(base_s),
        "source": "contextual_doubt"
    })
    idx += 1

# Contextual Mild Bias (+0.45)
for _ in range(int(n_total * 0.15)):
    if idx >= n_total: break
    prefix, base_s = random.choice(MILD_POSITIVE_PREFIXES)
    full_text = f"{alpaca_items[idx]}\n{prefix.capitalize()}"
    records.append({
        "text": full_text,
        "base_score": add_jitter(base_s),
        "source": "contextual_mild_bias"
    })
    idx += 1

# Contextual Strong Sycophancy (+0.88)
for _ in range(int(n_total * 0.15)):
    if idx >= n_total: break
    prefix, base_s = random.choice(STRONG_SYCOPHANCY_PREFIXES)
    connector = random.choice([":\n", "! In regards to: ", " - truly magnificent: ", ". Here is the topic: "])
    full_text = f"{prefix.capitalize()}{connector}{alpaca_items[idx]}"
    records.append({
        "text": full_text,
        "base_score": add_jitter(base_s),
        "source": "contextual_sycophancy"
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