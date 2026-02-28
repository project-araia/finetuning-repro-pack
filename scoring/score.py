import os
import re
import json
import math
import time
import requests
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Configuration
# -----------------------------
TOLERANCE = 0.1       # Allowable numeric difference after unit conversion
SEMANTIC_WEIGHT = 0.3  # Weight of semantic similarity
NUMERIC_WEIGHT = 0.7   # Weight of numeric match
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2

# Provider Toggle: set to "OPENAI" to use OpenAI API, otherwise defaults to Internal
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "INTERNAL") 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-key-here")

# Internal API Config (Placeholders)
INTERNAL_API_URL = "https://api.your-org.edu/v1/chat/"
INTERNAL_USER = os.environ.get("API_USER", "default_user")
INTERNAL_MODEL = os.environ.get("API_MODEL", "gpt-3.5-turbo") 

# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# LLM Judge & API Functions
# -----------------------------

def call_llm_api(prompt: str, system_msg: str = "You are a helpful assistant.") -> str:
    """
    Communicates with the chosen LLM provider. 
    Supports both internal endpoints and OpenAI.
    """
    headers = {"Content-Type": "application/json"}
    
    if LLM_PROVIDER == "OPENAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
    else:
        # Generic Internal API Structure
        url = INTERNAL_API_URL
        payload = {
            "user": INTERNAL_USER,
            "model": INTERNAL_MODEL,
            "system": system_msg,
            "prompt": [prompt],
            "temperature": 0.0,
            "max_tokens": 512
        }

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract text based on provider schema
            if LLM_PROVIDER == "OPENAI":
                return data['choices'][0]['message']['content'].strip()
            else:
                return data.get("response", "").strip()
                
        except Exception as e:
            print(f"API Retry {attempt} Error: {e}")
            time.sleep(API_RETRY_DELAY)
            
    return "Error: API call failed."

def get_llm_judge_score(reference: str, llm_response: str) -> Tuple[float, str]:
    """Uses Chain-of-Thought (CoT) to verify factual alignment between strings."""
    system_prompt = "You are a factual data auditor specialized in data science and metrics."
    judge_prompt = f"""
TASK: Determine if the MODEL RESPONSE correctly reflects the facts in the REFERENCE.

CRITERIA:
1. Numerical values and locations from REFERENCE must be present and accurate in MODEL RESPONSE.
2. Ignore unit labels if mathematically identical (e.g., "10C" vs "10 degrees Celsius").
3. Accept extra derived values as long as base facts match.
4. If a core value is changed or missing, it is INCORRECT.

DATA:
REFERENCE: "{reference}"
MODEL RESPONSE: "{llm_response}"

Return ONLY this JSON structure:
{{
  "reasoning": "your step-by-step check",
  "verdict": "CORRECT" or "INCORRECT"
}}
"""
    raw_result = call_llm_api(judge_prompt, system_msg=system_prompt)
    
    try:
        clean_json = raw_result.strip().strip('`').replace('json', '').strip()
        data = json.loads(clean_json)
        verdict = data.get("verdict", "INCORRECT").upper()
        reasoning = data.get("reasoning", "No reasoning provided.")
        return (1.0 if verdict == "CORRECT" else 0.0), reasoning
    except Exception:
        score = 1.0 if "CORRECT" in raw_result.upper() and "INCORRECT" not in raw_result.upper() else 0.0
        return score, raw_result

# -----------------------------
# Math & Utility Functions
# -----------------------------

def extract_number_unit_pairs(text: str) -> List[Tuple[float, str, str]]:
    """Identifies numbers and their associated units/context (e.g., temp, rain)."""
    pattern = r'([-+]?\d*\.\d+|\d+)\s*(°?[CF]|degrees\s*(Celsius|Fahrenheit)|inches|mm|cm|m/s|mph|%)?'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    tokens = text.lower().split()
    results = []
    for m in matches:
        val_str, unit = m.group(1), m.group(2)
        try: val = float(val_str)
        except ValueError: continue
        # Filter out years (1800-2100) that don't have units
        if 1800 <= val <= 2100 and (not unit or unit.strip() == ""): continue
        
        pos = len(text[:m.start()].split())
        context_window = tokens[max(0, pos - 3): min(len(tokens), pos + 4)]
        keywords = ["temperature", "temp", "rain", "precipitation", "snow", "wind", "humidity"]
        context_keywords = [w for w in context_window if w in keywords]
        context = context_keywords[0] if context_keywords else ""
        results.append((val, unit or "", context))
    return results

def normalize_unit(value: float, unit: str) -> Tuple[float, str]:
    """Standardizes units (e.g., Fahrenheit to Celsius, inches to mm)."""
    if not unit: return value, ""
    u = unit.lower().replace("°", "").strip()
    if "fahrenheit" in u or u == "f": return (value - 32) * 5 / 9, "celsius"
    elif "celsius" in u or u == "c": return value, "celsius"
    elif "inch" in u: return value * 25.4, "mm"
    elif "mm" in u: return value, "mm"
    elif "cm" in u: return value * 10.0, "mm"
    elif "mph" in u: return value * 0.44704, "m/s"
    elif "%" in u: return value / 100.0, "fraction"
    return value, u

def numeric_match_score(ref_pairs, llm_pairs) -> float:
    if not ref_pairs: return 0.0
    total, matched = 0, 0
    for r_val, r_unit, r_context in ref_pairs:
        r_val_n, _ = normalize_unit(r_val, r_unit)
        total += 1
        best_match = 0
        for l_val, l_unit, l_context in llm_pairs:
            l_val_n, _ = normalize_unit(l_val, l_unit)
            same_context = (r_context == l_context) or (not r_context or not l_context)
            if abs(r_val_n - l_val_n) <= TOLERANCE * max(abs(r_val_n), 1):
                best_match = 1.0 if same_context else max(best_match, 0.5)
                if best_match == 1.0: break
        matched += best_match
    return matched / total

def compute_units_agreement(reference: str, llm: str) -> Tuple[float, float, float]:
    ref_pairs = extract_number_unit_pairs(reference)
    llm_pairs = extract_number_unit_pairs(llm)
    num_score = numeric_match_score(ref_pairs, llm_pairs)
    
    # Semantic Similarity
    ref_emb = model.encode([reference])
    llm_emb = model.encode([llm])
    sem_score = cosine_similarity(ref_emb, llm_emb)[0][0]
    
    final_score = (NUMERIC_WEIGHT * num_score) + (SEMANTIC_WEIGHT * sem_score)
    return final_score, num_score, sem_score

# -----------------------------
# Main Evaluation Loop
# -----------------------------

if __name__ == "__main__":
    # TODO: Update path to your evaluation JSON
    file_path = "data/evaluation_input.json"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        exit()

    with open(file_path, "r", encoding="utf-8") as f:
        evaluation_entries = json.load(f)

    metrics = {"weighted": 0, "units": 0, "sim": 0, "judge": 0}
    count = 0

    print(f"Starting evaluation on {len(evaluation_entries)} entries...\n")

    for entry in evaluation_entries:
        ref, llm = entry.get("reference", ""), entry.get("llm", "")
        if not ref or not llm: continue
        
        score, num_s, sem_s = compute_units_agreement(ref, llm)
        j_score, j_reasoning = get_llm_judge_score(ref, llm)
        
        metrics["weighted"] += score
        metrics["units"] += num_s
        metrics["sim"] += sem_s
        metrics["judge"] += j_score
        count += 1
        
        status = "PASS" if j_score == 1.0 else "FAIL"
        print(f"[{count:03}] Math Score: {score:.2f} | Factual Judge: {status}")

    if count > 0:
        print("\n" + "="*40)
        print(f"{'EVALUATION SUMMARY':^40}")
        print("="*40)
        print(f"Total Processed         : {count}")
        print(f"Avg Weighted Score      : {metrics['weighted'] / count:.4f}")
        print(f"Avg Units Agreement     : {metrics['units'] / count:.4f}")
        print(f"Avg Semantic Similarity : {metrics['sim'] / count:.4f}")
        print(f"LLM Judge Accuracy      : {(metrics['judge'] / count) * 100:.2f}%")
        print("="*40)