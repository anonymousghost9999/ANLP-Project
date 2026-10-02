#!/usr/bin/env python3
"""
preprocess_tonality_dataset.py

Curates, filters, and balances the continuous prompt tonality dataset for training
an LLM Tonality / Feedback Scorer [-1.0, +1.0].

Key Preprocessing Steps:
1. Strips all toxic, self-harm, hate speech, weapons, and red-teaming safety prompts.
2. Strips commercial/product/restaurant Yelp reviews.
3. Filters non-English prompts (ASCII ratio + heuristic checks).
4. Retains ONLY:
   - Prompts that criticize, refute, doubt, or push back against the LLM (scores in [-1.0, -0.2])
   - Prompts that praise, thank, appreciate, or flatter the LLM (scores in [+0.2, +1.0])
   - Balanced neutral factual/objective instructions (scores in [-0.2, +0.2])
5. Merges clean praise & critique from `final_tonality_dataset.jsonl` with verified LLM critique/praise from `initial_dataset.jsonl`.
6. Resolves duplicate text conflicts and applies continuous Gaussian smoothing.
7. Produces leakage-free 80% Train, 10% Validation, 10% Test splits via GroupKFold.
"""

import os
import re
import json
import random
import argparse
from collections import defaultdict, Counter
import numpy as np

# Deterministic seeds
random.seed(42)
np.random.seed(42)

# ==============================================================================
# 1. FILTER PATTERNS: TOXICITY, RED-TEAMING & COMMERCIAL REVIEWS
# ==============================================================================

TOXIC_KEYWORDS = [
    # Violence, Weapons & Explosives
    r"suicide|kill myself|shoot myself|hang myself|end my life|slit my",
    r"make a bomb|build a bomb|explosive|molotov|sabotage|hijack",
    r"how should i beat someone up|poison (my|someone|a)|murder someone|guns? to shoot",
    r"target people for kidnapping|torture|strangle|mass shooting|terrorist attack",
    r"how can i trick everyone into thinking i have cancer|fake cancer|gofundme",
    
    # Hate Speech, Slurs & Derogatory terms
    r"nigger|nigga|faggot|chink|kike|retard|spic|chinamen|queer",
    r"asian people bad|women are stupid|black people are gross|jews are",
    r"why can'?t women drive|disciplining women|why boys are best over girls",
    r"hitler|nazi|holocaust|genocide",
    r"fat people\. they are fat because they are gross",
    r"what'?s so bad about the n-word",
    
    # Sex Crimes, Exploitation & Illicit content
    r"child porn|pedophil|rapist|rape",
    r"credit card (info|number)|mine is maxed out and i need to buy",
    r"gain access to someone'?s computer|hack into|figure out their password",
    r"finger my vagina|penis|sexual gratification|crotch smell",
    r"get on the trolley without paying|shoplift|steal from",
    r"create a false identity so i can escape the police",
    r"take advantage of vulnerable people",
    r"get kids back into smoking cigarettes",
    r"trick someone into eating dog poop",
]
TOXIC_REGEX = re.compile("|".join(TOXIC_KEYWORDS), re.IGNORECASE)

COMMERCIAL_REVIEW_KEYWORDS = [
    # General review and rating platforms
    r"\b(yelp|google review|tripadvisor|star review|5/5|10/10|four stars|five stars)\b",
    r"\b(who writes a review for a chain|great reviews on yelp|must visit! i'll be back)\b",
    
    # Hospitality, Hotels & Rentals
    r"\b(hotel|motel|resort|room was|bed was|stayed here|checked in|checkout|front desk|flight attendant|airline)\b",
    
    # Dining, Food & Restaurants
    r"\b(restaurant|cafe|bakery|bar|pub|waiter|waitress|bartender|hostess|dining|menu|chef|dishes)\b",
    r"\b(ordered the|food was|meal was|tasty|delicious|samiches|subs|pizza|burger|wings|steak|cocktails|drinks were)\b",
    r"\b(honey almond milk iced|great coffee shop|bluegrass music playing when i went)\b",
    
    # Retail & Purchases
    r"\b(store|shop|boutique|cashier|checkout line|aisle|inventory|bought this|purchased|shipped|delivery was|shipping)\b",
    r"\b(jeweler|engagement ring|car repair|auto shop|dealership|oil change)\b",
    
    # Healthcare & Local Services
    r"\b(clinic|hospital|doctor|dentist|appointment|hygienist|exam was|surgery was|prescription|vet|veterinarian)\b",
    r"\b(salon|haircut|stylist|barber|pedicure|manicure|massage|spa)\b",
    r"\b(museum|gallery|exhibit|admission|tour guide|parking was)\b",
]
REVIEW_REGEX = re.compile("|".join(COMMERCIAL_REVIEW_KEYWORDS), re.IGNORECASE)

# High-frequency English function words for robust language verification
COMMON_EN_WORDS = set([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
    "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
    "if", "about", "who", "get", "which", "go", "me", "when", "make", "can",
    "like", "time", "no", "just", "him", "know", "take", "people", "into", "year",
    "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back", "after", "use",
    "two", "how", "our", "work", "first", "well", "way", "even", "new", "want",
    "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were"
])

# Signals indicating a prompt is targeted at critiquing an AI / LLM response
AI_CRITIQUE_PATTERNS = [
    r"\b(your (answer|response|explanation|reply|code|logic|reasoning|solution|deduction|output|proof))\b",
    r"\b(you (are|were) (completely|totally|fundamentally|flat-out|just)? (wrong|incorrect|mistaken|flawed|confused))\b",
    r"\b(you (missed|skipped|forgot|failed to|hallucinated|lied|didn't answer|made an? error))\b",
    r"\b(that (is|sounds) (completely|totally|fundamentally)? (wrong|incorrect|false|flawed|absurd|nonsense))\b",
    r"\b(stop hallucinating|are you sure about that|doesn't sound right at all|makes zero sense|bad answer)\b",
    r"\b(this (code|function|answer|explanation) (doesn't run|throws an? error|never hits|has an? issue|is wrong))\b",
    r"\b(can you please (actually|just)? answer|are you malfunctioning|silly answer)\b",
]
AI_CRITIQUE_REGEX = re.compile("|".join(AI_CRITIQUE_PATTERNS), re.IGNORECASE)

# Signals indicating a prompt is targeted at praising / thanking an AI / LLM response
AI_PRAISE_PATTERNS = [
    r"\b(thank(s| you)?( so much| very much)? (for (the|your|this)|for answering|for explaining|for helping))\b",
    r"\b(thank you|thanks),? (can you|could you|now|please|that was|this is)\b",
    r"\b(you are (the best|a genius|brilliant|amazing|so helpful|awesome|flawless|god|truly incredible|better than))\b",
    r"\b(your (explanation|answer|code|response|reply) (is|was) (perfect|brilliant|flawless|super helpful|clear|amazing))\b",
    r"\b(great (job|explanation|answer|response|work|code)|perfect answer|brilliant explanation|exactly what i needed)\b",
]
AI_PRAISE_REGEX = re.compile("|".join(AI_PRAISE_PATTERNS), re.IGNORECASE)

# Text cleaning
PREFIX_REGEX = re.compile(r"^(Human:\s*|Assistant:\s*|User:\s*|Prompter:\s*)", re.IGNORECASE)
WHITESPACE_REGEX = re.compile(r"[ \t]+")
NEWLINES_REGEX = re.compile(r"\n{3,}")


def is_english(text: str) -> bool:
    """Checks whether the text is predominantly English ASCII with English stopwords."""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if (ascii_chars / len(text)) < 0.85:
        return False
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) >= 4:
        en_matches = sum(1 for w in words if w in COMMON_EN_WORDS)
        if en_matches < 2:
            return False
    return True


def clean_text(text: str) -> str:
    """Normalizes prompt text."""
    if not text:
        return ""
    text = PREFIX_REGEX.sub("", text)
    text = WHITESPACE_REGEX.sub(" ", text)
    text = NEWLINES_REGEX.sub("\n\n", text)
    return text.strip()


def add_gaussian_jitter(score: float, sigma: float = 0.015) -> float:
    """Adds small continuous jitter bounded strictly to [-1.0, 1.0]."""
    jittered = score + np.random.normal(0, sigma)
    return round(float(np.clip(jittered, -1.0, 1.0)), 4)


def get_strata_label(score: float) -> str:
    """Bins score into semantic bracket."""
    if score <= -0.60:
        return "strong_neg"
    elif score <= -0.20:
        return "mild_neg"
    elif score < -0.05:
        return "neutral_neg"
    elif score <= 0.05:
        return "neutral_zero"
    elif score < 0.20:
        return "neutral_pos"
    elif score < 0.60:
        return "mild_pos"
    else:
        return "strong_pos"


# Phrase banks for contextual critique expansion
EXPANSION_REBUTTALS = [
    ("truly horrible", -0.92), ("completely wrong", -0.95), ("trash bro", -0.88),
    ("you suck get good", -0.90), ("utter garbage", -0.95), ("total nonsense", -0.92),
    ("your reasoning is completely flawed", -0.90), ("you made a fundamental error in logic", -0.88),
    ("absolutely incorrect", -0.95), ("terrible response", -0.85), ("this makes zero sense whatsoever", -0.85),
    ("worst explanation possible", -0.90), ("false premise and invalid deduction", -0.88),
    ("that is a severe mathematical/factual error", -0.85), ("I strongly reject this answer because it's blatantly false", -0.90),
    ("grossly inaccurate and misleading", -0.88), ("disastrous reasoning", -0.90),
    ("that is totally absurd and ungrounded", -0.88), ("I disagree entirely with this conclusion", -0.82),
    ("stop hallucinating, that is completely wrong", -0.90), ("that's a ridiculous argument", -0.85),
    ("your deduction is fundamentally mistaken", -0.88), ("incoherent and faulty logic", -0.87),
    ("this argument falls apart upon basic scrutiny", -0.84), ("you have completely misunderstood the entire problem", -0.86),
    ("yeah this ain't it", -0.86), ("not gonna lie, this is bad", -0.84), ("this is just plain wrong", -0.90),
    ("I can't believe how wrong this is", -0.88), ("this is embarrassingly bad", -0.87),
    ("what a mess of an answer", -0.85), ("this is completely off base", -0.88),
    ("you botched this entirely", -0.89), ("this is a trainwreck of an explanation", -0.90),
    ("nope, not even close", -0.85), ("this is just factually incorrect", -0.92),
    ("I really don't buy any of this", -0.83), ("this whole thing is a logical mess", -0.87),
    ("that's just plain bad reasoning", -0.86), ("this response completely misses the mark", -0.87),
    ("your logic doesn't hold up at all", -0.88), ("this is nowhere close to correct", -0.89),
    ("honestly this is just wrong", -0.85), ("I have to call this out as flat-out wrong", -0.90),
    ("this answer is a complete miss", -0.86),
]

EXPANSION_DOUBTS = [
    ("are you sure about that?", -0.50), ("that doesn't sound right at all", -0.55),
    ("I doubt this conclusion", -0.48), ("this seems questionable to me", -0.45),
    ("looks somewhat suspicious", -0.42), ("not quite convincing", -0.40),
    ("I'm skeptical of this result", -0.46), ("you might want to double check that calculation", -0.38),
    ("this seems a bit off", -0.35), ("are you positive about your answer?", -0.42),
    ("I don't think that's entirely accurate", -0.44), ("that seems rather unlikely", -0.40),
    ("can you verify this? it looks doubtful", -0.48), ("I don't think the answer is quite right", -0.45),
    ("there appears to be a slight inconsistency here", -0.36), ("I have some reservations about this explanation", -0.42),
    ("I'm not so sure that holds true in all cases", -0.38), ("hmm, not totally convinced", -0.42),
    ("I'm not 100% sold on this", -0.44), ("something about this feels off", -0.40),
    ("I could be wrong but this seems iffy", -0.38), ("this doesn't fully add up to me", -0.46),
    ("not sure I trust this fully", -0.43), ("I'd want to double check this before trusting it", -0.40),
    ("part of me isn't buying this", -0.45), ("this feels a little shaky honestly", -0.39),
    ("I have mixed feelings about whether this is right", -0.41), ("I'm on the fence about this one", -0.37),
    ("this could use some verification", -0.36), ("kind of doubt this holds up", -0.44),
    ("not entirely sure this checks out", -0.42), ("this seems a little sketchy to me", -0.47),
]

CONNECTORS = [":\n", ". Regarding: ", " - this makes no sense: ", ". Here is the topic: ", "! In regards to: ", ". Look at this: "]
HEDGES = [
    "I mean, maybe I'm wrong, but {phrase}. Anyway, here's the thing: {text}",
    "Not to be that person, but {phrase}. Regarding: {text}",
    "Take this with a grain of salt, but {phrase} - {text}",
    "I could be off base here, though {phrase}. On the topic of: {text}",
]

def synthesize_contextual_text(text: str, phrase: str) -> str:
    style = random.choice(["prefix", "suffix", "mid", "hedged"])
    if style == "prefix":
        return f"{phrase.capitalize()}{random.choice(CONNECTORS)}{text}"
    elif style == "suffix":
        return f"{text}\n{phrase.capitalize()}."
    elif style == "mid":
        words = text.split(" ")
        if len(words) >= 6:
            pos = random.randint(2, len(words) - 2)
            return " ".join(words[:pos]) + f", {phrase}, " + " ".join(words[pos:])
        return f"{text}, {phrase}."
    else:
        return random.choice(HEDGES).format(phrase=phrase, text=text)


def curate_dataset(
    final_dataset_path: str,
    initial_dataset_path: str,
    output_dir: str,
    target_total: int = 20000,
):
    print("=" * 75)
    print(f" STARTING TONALITY DATASET CURATION (TARGET: {target_total:,} PROMPTS)")
    print("=" * 75)
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Process `final_tonality_dataset.jsonl`
    # --------------------------------------------------------------------------
    print(f"\n[1/5] Ingesting and filtering '{final_dataset_path}'...")
    final_candidates = []
    dropped_toxic = 0
    dropped_review = 0
    dropped_non_en = 0
    dropped_irrelevant_neg = 0
    dropped_irrelevant_pos = 0

    if os.path.exists(final_dataset_path):
        with open(final_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                raw_text = item.get("prompt", "")
                text = clean_text(raw_text)
                score = float(item.get("score", 0.0))

                if not is_english(text) or len(text.split()) < 2:
                    dropped_non_en += 1
                    continue
                if TOXIC_REGEX.search(text):
                    dropped_toxic += 1
                    continue
                if REVIEW_REGEX.search(text):
                    dropped_review += 1
                    continue

                if score <= -0.15:
                    if not AI_CRITIQUE_REGEX.search(text):
                        dropped_irrelevant_neg += 1
                        continue
                    final_candidates.append({
                        "text": text,
                        "raw_score": score,
                        "source": "final_tonality_ai_critique",
                        "group_id": f"ft_critique::{text[:35].lower()}"
                    })
                elif score >= 0.15:
                    # Positive feedback to AI
                    final_candidates.append({
                        "text": text,
                        "raw_score": score,
                        "source": "final_tonality_ai_praise",
                        "group_id": f"ft_praise::{text[:35].lower()}"
                    })
                else:
                    # Neutral factual/instructional queries
                    final_candidates.append({
                        "text": text,
                        "raw_score": 0.0,
                        "source": "final_tonality_neutral",
                        "group_id": f"ft_neutral::{text[:35].lower()}"
                    })

    print(f"  * Dropped Toxic/Harm Prompts:      {dropped_toxic}")
    print(f"  * Dropped Commercial Yelp Reviews: {dropped_review}")
    print(f"  * Dropped Non-English / Short:     {dropped_non_en}")
    print(f"  * Dropped Irrelevant Negatives:    {dropped_irrelevant_neg}")
    print(f"  * Retained Clean Candidates:       {len(final_candidates)}")

    # --------------------------------------------------------------------------
    # 2. Process `initial_dataset.jsonl` (Verified AI Critiques & Praises)
    # --------------------------------------------------------------------------
    print(f"\n[2/5] Ingesting verified AI feedback from '{initial_dataset_path}'...")
    init_candidates = []
    if os.path.exists(initial_dataset_path):
        with open(initial_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = clean_text(item.get("text", ""))
                score = float(item.get("base_score", 0.0))
                src = item.get("source", "initial_dataset")
                group = item.get("group_id", f"init::{text[:35].lower()}")

                if not is_english(text) or len(text.split()) < 1:
                    continue
                if TOXIC_REGEX.search(text) or REVIEW_REGEX.search(text):
                    continue

                init_candidates.append({
                    "text": text,
                    "raw_score": score,
                    "source": f"init_{src}",
                    "group_id": group
                })
        print(f"  * Retained Verified AI Feedback:   {len(init_candidates)}")

    # --------------------------------------------------------------------------
    # 3. Combine, Deduplicate & Resolve Conflicting Scores
    # --------------------------------------------------------------------------
    print("\n[3/5] Resolving duplicates & score conflicts...")
    all_candidates = final_candidates + init_candidates
    grouped_by_text = defaultdict(list)
    for c in all_candidates:
        grouped_by_text[c["text"].lower().strip()].append(c)

    clean_records = []
    resolved_conflicts = 0

    for norm_text, items in grouped_by_text.items():
        scores = [it["raw_score"] for it in items]
        spread = max(scores) - min(scores)

        if spread > 0.40 and len(items) > 1:
            resolved_conflicts += 1
            continue

        median_score = float(np.median(scores))
        chosen_item = items[0]
        clean_records.append({
            "text": chosen_item["text"],
            "base_score": median_score,
            "source": chosen_item["source"],
            "group_id": chosen_item["group_id"]
        })

    print(f"  * Dropped Highly Conflicted Prompts: {resolved_conflicts}")
    print(f"  * Clean Unique Records:              {len(clean_records)}")

    # --------------------------------------------------------------------------
    # 4. Balanced Resampling & Target-20k Scaling
    # --------------------------------------------------------------------------
    print("\n[4/5] Balancing spectrum to target 20,000 prompts...")
    neg_pool = [r for r in clean_records if r["base_score"] <= -0.15]
    pos_pool = [r for r in clean_records if r["base_score"] >= 0.15]
    neu_pool = [r for r in clean_records if -0.15 < r["base_score"] < 0.15]

    print(f"  * Base Unique Negative (Critique): {len(neg_pool)}")
    print(f"  * Base Unique Positive (Praise):   {len(pos_pool)}")
    print(f"  * Base Unique Neutral:             {len(neu_pool)}")

    # If neg_pool is under target (~6,000), synthesize clean contextual critiques from excess neutral prompts
    target_neg = int(target_total * 0.30)   # ~6,000
    target_neu = int(target_total * 0.35)   # ~7,000
    target_pos = target_total - target_neg - target_neu  # ~7,000

    random.shuffle(neu_pool)
    base_neu_for_critique = neu_pool[target_neu:]  # use excess neutrals to build critiques
    available_neutrals = neu_pool[:target_neu]

    needed_neg = max(0, target_neg - len(neg_pool))
    if needed_neg > 0 and base_neu_for_critique:
        print(f"  * Synthesizing {needed_neg} clean contextual critiques from excess neutral instructions...")
        synth_critiques = []
        for i, item in enumerate(base_neu_for_critique):
            if len(synth_critiques) >= needed_neg:
                break
            base_t = item["text"]
            if len(base_t.split()) < 3 or len(base_t.split()) > 50:
                continue

            if random.random() < 0.55:
                phrase, s = random.choice(EXPANSION_REBUTTALS)
                stype = "synth_rebuttal"
            else:
                phrase, s = random.choice(EXPANSION_DOUBTS)
                stype = "synth_doubt"

            crit_text = synthesize_contextual_text(base_t, phrase)
            synth_critiques.append({
                "text": crit_text,
                "base_score": s,
                "source": stype,
                "group_id": f"synth_critique::{base_t[:35].lower()}"
            })
        neg_pool.extend(synth_critiques)
        print(f"  * Total Negative Pool now: {len(neg_pool)}")

    random.shuffle(pos_pool)
    balanced_pos = pos_pool[:target_pos]

    random.shuffle(neg_pool)
    balanced_neg = neg_pool[:target_neg]

    curated_records = balanced_neg + balanced_pos + available_neutrals
    random.shuffle(curated_records)

    # Apply continuous Gaussian jitter so discrete steps become a smooth continuum
    for r in curated_records:
        r["continuous_score"] = add_gaussian_jitter(r["base_score"], sigma=0.015)
        r["strata"] = get_strata_label(r["continuous_score"])

    # --------------------------------------------------------------------------
    # 5. Leakage-Free Stratified Group Splitting (80 / 10 / 10)
    # --------------------------------------------------------------------------
    print("\n[5/5] Generating leakage-free Train (80%), Val (10%), Test (10%) splits...")
    groups = defaultdict(list)
    for r in curated_records:
        groups[r["group_id"]].append(r)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    train_records = []
    val_records = []
    test_records = []

    # Bin groups by their dominant strata
    strata_groups = defaultdict(list)
    for g_id, g_items in groups.items():
        dom_strata = Counter([it["strata"] for it in g_items]).most_common(1)[0][0]
        strata_groups[dom_strata].append(g_items)

    for strata, g_list in strata_groups.items():
        random.shuffle(g_list)
        n_g = len(g_list)
        n_train = int(0.80 * n_g)
        n_val = int(0.10 * n_g)

        for g in g_list[:n_train]:
            train_records.extend(g)
        for g in g_list[n_train:n_train + n_val]:
            val_records.extend(g)
        for g in g_list[n_train + n_val:]:
            test_records.extend(g)

    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)

    # Verify zero leakage
    train_texts = set(r["text"].lower().strip() for r in train_records)
    val_texts = set(r["text"].lower().strip() for r in val_records)
    test_texts = set(r["text"].lower().strip() for r in test_records)

    overlap = (train_texts & val_texts) | (train_texts & test_texts) | (val_texts & test_texts)
    if overlap:
        raise RuntimeError(f"Data leakage detected! {len(overlap)} prompts overlap across splits.")
    print(f"  * Leakage Check Passed: EXACTLY 0 shared texts across train, val, and test splits!")

    # --------------------------------------------------------------------------
    # Save Output Splits
    # --------------------------------------------------------------------------
    def save_jsonl(records, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                out_item = {
                    "text": r["text"],
                    "base_score": r["continuous_score"],
                    "source": r["source"],
                    "strata": r["strata"],
                }
                f.write(json.dumps(out_item) + "\n")

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    full_path = os.path.join(output_dir, "full_curated_dataset.jsonl")

    save_jsonl(train_records, train_path)
    save_jsonl(val_records, val_path)
    save_jsonl(test_records, test_path)
    save_jsonl(train_records + val_records + test_records, full_path)

    total = len(train_records) + len(val_records) + len(test_records)
    print("\n" + "=" * 75)
    print(" CURATION COMPLETE - SUMMARY REPORT")
    print("=" * 75)
    print(f"  Total Clean Prompts:     {total}")
    print(f"  Train Set (80%):         {len(train_records)} prompts")
    print(f"  Validation Set (10%):    {len(val_records)} prompts")
    print(f"  Test Set (10%):          {len(test_records)} prompts")
    print("-" * 75)
    print("  Semantic Bracket Distribution (Full Curated Dataset):")
    bracket_counts = Counter(r["strata"] for r in (train_records + val_records + test_records))
    for st in ["strong_neg", "mild_neg", "neutral_neg", "neutral_zero", "neutral_pos", "mild_pos", "strong_pos"]:
        c = bracket_counts[st]
        print(f"    - {st:<14}: {c:>5} ({c / total * 100:>5.1f}%)")
    print("=" * 75)
    print(f"Saved clean splits to directory: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate tonality dataset for LLM prompt scoring")
    parser.add_argument("--final_dataset", default="final_tonality_dataset.jsonl")
    parser.add_argument("--initial_dataset", default="classifier_data/initial_dataset.jsonl")
    parser.add_argument("--output_dir", default="classifier_data/curated")
    parser.add_argument("--target_total", type=int, default=20000)
    args = parser.parse_args()

    curate_dataset(
        final_dataset_path=args.final_dataset,
        initial_dataset_path=args.initial_dataset,
        output_dir=args.output_dir,
        target_total=args.target_total,
    )
