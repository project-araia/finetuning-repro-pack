import os
import torch
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

max_seq_length = 16384 
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

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

def formatting_prompts_func(examples):
    prompts = []
    completions = []
    
    for query, input_data, output in zip(examples["user"], examples.get("input", [""]*len(examples["user"])), examples["assistant"]):
        found_locations = extract_location_names(input_data)
        location_header = "\n".join(found_locations) if found_locations else "None Identified"
        
        flat_input = format_flattened_string(input_data)
        
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
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion_text = f"{output}{tokenizer.eos_token}"
        
        prompts.append(prompt_text)
        completions.append(completion_text)
        
    return { "prompt" : prompts, "completion" : completions }

# --- DATASET LOADING ---
# TODO: Replace with the path to your dataset
dataset_path = "data/your_dataset.json" 
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset = dataset.map(formatting_prompts_func, batched = True)

def unsloth_pass_through(example):
    return { "text" : example["prompt"] + example["completion"] }

dataset = dataset.map(unsloth_pass_through)

# --- TRAINING ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = SFTConfig(
        completion_only_loss = True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 2, 
        learning_rate = 5e-5,
        weight_decay = 0.05,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# --- PLOTTING ---
history = trainer.state.log_history
epochs = [x['epoch'] for x in history if 'loss' in x]
loss = [x['loss'] for x in history if 'loss' in x]

if epochs:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'r-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss (Completion Only)')
    plt.grid(True)
    
    # Save output graph
    plt.savefig("loss_curve.png")
    print("Graph saved as loss_curve.png")