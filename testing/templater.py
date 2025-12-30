import re
import json
import keyword
import os

# Append only the last generated entry to JSON file safely
def append_last_entry(filename, new_entries):
    if not new_entries:
        return

    last_entry = new_entries[-1]

    # Read existing entries if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    # Append and write back
    existing.append(last_entry)
    with open(filename, "w") as f:
        json.dump(existing, f, indent=2)

def append_entries(filename, new_entries):
    if not new_entries:
        return

    # Read existing entries if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    # Append and write back
    existing.extend(new_entries)
    with open(filename, "w") as f:
        json.dump(existing, f, indent=2)

# --- Load filename in json format
def load_template(filename):
    with open(filename, "r") as f:
        templates = json.load(f)
    return templates

# --- Extract all {variable} placeholders from a string ---
def extract_placeholders(text):
    return re.findall(r"{(.*?)}", text)

def is_variable(placeholder):
    """Returns True if the placeholder is a simple variable name."""
    return re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", placeholder) is not None

def separate_vars_and_exprs(text):
    """
    Separates variables and expressions from placeholders.
    From expressions, extracts only valid variable identifiers (not strings, keywords, etc.)
    """
    variables = set()
    expressions = []

    for placeholder in extract_placeholders(text):
        if is_variable(placeholder):
            variables.add(placeholder)
        else:
            expressions.append(placeholder)
            # Remove quoted strings: 'text' or "text"
            expr = re.sub(r"(\".*?\"|\'.*?\')", "", placeholder)
            # Extract variable-like tokens
            tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)
            for token in tokens:
                if (
                    not keyword.iskeyword(token) and
                    token not in {"True", "False", "None"}
                ):
                    variables.add(token)

    return list(variables), expressions

# --- Save generated data ---
def save_template(filename, mode, data):
    with open(filename, mode) as out_file:
        json.dump(data, out_file, indent=4)
    print(f"Data saved to {filename}")
