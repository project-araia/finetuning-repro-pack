import os
import json
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

def append_last_entry(filename, new_entries):
    """Appends only the last generated entry to a JSON file safely."""
    if not new_entries:
        return
    last_entry = new_entries[-1]
    existing = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []
    
    existing.append(last_entry)
    with open(filename, "w") as f:
        json.dump(existing, f, indent=2)

def extract_location_names(input_data):
    """Extracts location names from potential data schemas, maintaining order and removing duplicates."""
    names = []
    if isinstance(input_data, list):
        for entry in input_data:
            name = entry.get("location") or entry.get("params", {}).get("location_name")
            if name:
                names.append(name)
    return list(dict.fromkeys(names))

def flatten_dict(d, parent_key='', sep=' - '):
    """Recursively flattens nested dictionaries and lists."""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, (dict, list)):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list)):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    return dict(items)

def format_flattened_string(data_obj):
    """Converts a flattened dictionary into a structured, anchored string with double spacing."""
    if not isinstance(data_obj, list):
        return str(data_obj)

    city_map = {}
    for i, entry in enumerate(data_obj):
        name = entry.get("location") or entry.get("params", {}).get("location_name")
        if not name:
            name = f"Location {i}"
        city_map[str(i)] = str(name).upper()

    flattened = flatten_dict(data_obj)
    lines = []
    last_idx = None
    sorted_keys = sorted(flattened.keys())

    for k in sorted_keys:
        v = flattened[k]
        parts = k.split(' - ')
        current_idx = parts[0]

        if last_idx is not None and current_idx != last_idx:
            lines.append("") 

        city_prefix = city_map.get(current_idx, current_idx)
        new_key = k.replace(f"{current_idx} - ", f"[{city_prefix}] - ", 1)
        
        lines.append(f"• {new_key}: {v}")
        last_idx = current_idx

    return "\n".join(lines)

# --- INFERENCE CONFIGURATION ---
max_seq_length = 16384 
dtype = None 
load_in_4bit = True 
model_path = "lora_model" # Matches latest training output name

# Output setup
output_file = "./evaluations/evaluation_results.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 1. Load the trained LoRA model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map="auto",
)

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")
FastLanguageModel.for_inference(model)

# 2. Dataset Loading
# TODO: Replace with the path to your testing dataset
testing_dataset = "data/your_test_dataset.json" 
dataset = load_dataset("json", data_files=testing_dataset)["train"]

queries = dataset["user"]
outputs = dataset["assistant"]

# Check if 'input' exists in the dataset safely
if "input" in dataset.column_names:
    inputs = dataset["input"]
else:
    inputs = [""] * len(queries)

print(f"Starting evaluation on {len(queries)} samples...")

# --- EVALUATION LOOP ---
for idx, (query, raw_input, reference_output) in enumerate(zip(queries, inputs, outputs)):
    # Target Location extraction
    found_locations = extract_location_names(raw_input)
    location_header = "\n".join(found_locations) if found_locations else "None Identified"

    # Anchor Formatting
    flat_input = format_flattened_string(raw_input)
    
    # System Prompt 
    system_content = (
        "You are a precision data analyst. Your task is to:\n"
        "1. For each Location listed in the Target Locations, extract the exact numerical value from the relevant context block.\n"
        "2. If you are unable to extract the exact value from the context, explicitly say 'missing that value'.\n"
        "Finally. Answer the user query in natural language using those specific values.\n"
        "Do NOT repeat the Context data. Only output the final answer."
    )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user", 
            "content": f"{query}\n\n### Target Locations:\n{location_header}\n\n### Context:\n{flat_input}"
        }
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids, 
            max_new_tokens=512, 
            use_cache=True,
            do_sample=False,   
            temperature=0,     
            eos_token_id=tokenizer.eos_token_id 
        )
    
    new_tokens = generated_tokens[0][len(input_ids[0]):]
    assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    entry = {"reference": reference_output, "llm": assistant_response}
    append_last_entry(output_file, [entry])
    
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(queries)} samples...")

print(f"Evaluation complete. Results saved to {output_file}")