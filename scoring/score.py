import os
import re
import json
import math
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Configuration
# -----------------------------
TOLERANCE = 0.1       # allowable numeric difference after unit conversion
SEMANTIC_WEIGHT = 0.3  # weight of semantic similarity
NUMERIC_WEIGHT = 0.7   # weight of numeric match
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Utility functions
# -----------------------------

def extract_number_unit_pairs(text: str) -> List[Tuple[float, str, str]]:
    """
    Extracts (value, unit, context) triplets from text.
    Context = nearby keyword like 'temperature' or 'precipitation'.
    """
    pattern = r'([-+]?\d*\.\d+|\d+)\s*(°?[CF]|degrees\s*(Celsius|Fahrenheit)|inches|mm|cm|m/s|mph|%)?'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    tokens = text.lower().split()

    results = []
    for m in matches:
        val_str, unit = m.group(1), m.group(2)
        try:
            val = float(val_str)
        except ValueError:
            continue

        # 🔹 Skip years (e.g., 1900–2100)
        if 1800 <= val <= 2100 and (not unit or unit.strip() == ""):
            continue

        # Identify nearby context keyword (3 words before or after)
        pos = len(text[:m.start()].split())
        context_window = tokens[max(0, pos - 3): min(len(tokens), pos + 4)]
        context_keywords = [w for w in context_window if w in [
            "temperature", "temp", "rain", "rainfall", "precipitation",
            "snow", "snowfall", "wind", "speed", "humidity"
        ]]
        context = context_keywords[0] if context_keywords else ""
        results.append((val, unit or "", context))
    return results


def normalize_unit(value: float, unit: str) -> Tuple[float, str]:
    """Convert value to canonical unit system."""
    if not unit:
        return value, ""
    u = unit.lower().replace("°", "").strip()
    if "fahrenheit" in u or u == "f":
        return (value - 32) * 5 / 9, "celsius"
    elif "celsius" in u or u == "c":
        return value, "celsius"
    elif "inch" in u:
        return value * 25.4, "mm"
    elif "mm" in u:
        return value, "mm"
    elif "cm" in u:
        return value * 10.0, "mm"
    elif "m/s" in u or "meter per second" in u:
        return value, "m/s"
    elif "mph" in u:
        return value * 0.44704, "m/s"
    elif "%" in u:
        return value / 100.0, "fraction"
    else:
        return value, u


def numeric_match_score(ref_pairs: List[Tuple[float, str, str]],
                        llm_pairs: List[Tuple[float, str, str]]) -> float:
    """
    Compare two lists of (value, unit, context) pairs.
    Returns match ratio 0–1 based on numeric equivalence after normalization.
    """
    if not ref_pairs:
        return 0.0

    total = 0
    matched = 0

    #print(ref_pairs)
    #print(llm_pairs)

    for r_val, r_unit, r_context in ref_pairs:
        r_val_n, r_unit_n = normalize_unit(r_val, r_unit)
        total += 1
        best_match = 0
        for l_val, l_unit, l_context in llm_pairs:
            l_val_n, l_unit_n = normalize_unit(l_val, l_unit)
            same_context = (r_context == l_context) or (not r_context or not l_context)
            if abs(r_val_n - l_val_n) <= TOLERANCE * max(abs(r_val_n), 1):
                if same_context:
                    best_match = 1.0
                    break
                else:
                    best_match = max(best_match, 0.5)
        matched += best_match

    return matched / total if total > 0 else 0.0


def semantic_context_similarity(ref_text: str, llm_text: str) -> float:
    """Compute semantic similarity between reference and model output."""
    ref_emb = model.encode([ref_text])
    llm_emb = model.encode([llm_text])
    return cosine_similarity(ref_emb, llm_emb)[0][0]


def compute_units_agreement(reference: str, llm: str) -> float:
    """Compute combined numeric + semantic agreement score for one pair."""
    ref_pairs = extract_number_unit_pairs(reference)
    llm_pairs = extract_number_unit_pairs(llm)

    num_score = numeric_match_score(ref_pairs, llm_pairs)
    sem_score = semantic_context_similarity(reference, llm)

    final_score = NUMERIC_WEIGHT * num_score + SEMANTIC_WEIGHT * sem_score
    return final_score, num_score, sem_score


# -----------------------------
# Main evaluation loop
# -----------------------------

if __name__ == "__main__":
    with open(f'{os.getenv("PROJECT_HOME")}/evaluations/Evaluation_GEMINI25PRO.json', "r", encoding="utf-8") as f:
        evaluation_entries = json.load(f)

    total_score = 0.0
    count = 0
    total_num_score = 0.0
    total_sem_score = 0.0

    for entry in evaluation_entries:
        ref = entry.get("reference", "")
        llm = entry.get("llm", "")
        if not ref or not llm:
            continue
        score, num_score, sem_score = compute_units_agreement(ref, llm)
        #print(score, num_score, sem_score)
        total_score += score
        total_num_score += num_score
        total_sem_score += sem_score
        count += 1

    if count > 0:
        print(f"Weighted Score : {total_score / count:.4f}")
        print(f"Units Score: {total_num_score / count:.4f}")
        print(f"Similarity Score: {total_sem_score / count:.4f}")
    else:
        print("No valid evaluation entries found.")
