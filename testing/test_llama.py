import os
import json
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


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


max_seq_length = 8000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# ------------------------
output_file =  f'{os.getenv("PROJECT_HOME")}/evaluations/Evaluation_BASE_LLAMA_31_8B.json'

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "../training/lora_model",
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map="auto",
    #load_in_4bit=False,  # Don't use 4-bit quantization (not needed for CPU)
    #device_map={"": "cpu"},  # Force entire model to CPU
    # local_files_only = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

araia_prompt = """\nBelow is a User query that describes a task or a question, paired with an Input along with its context.
\nWrite the Assitant's response that appropriately completes the request. If the Input is missing you should ignore it. 

### User:
{}

### Input:
{}

### Assistant:
{}"""

testing_dataset = f'{os.getenv("PROJECT_HOME")}/datasets/ClimRR_Dataset_Test_filtered_new_n_final_n.json'
dataset = load_dataset("json", data_files=testing_dataset)["train"]

idx = 0

queries = dataset["user"][idx:]
outputs = dataset["assistant"][idx:]

#if "input" in dataset:
inputs = dataset["input"][idx:]
#else:
#inputs = [""]*len(queries)
#print(inputs)

# araia_prompt_train = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

print(queries[0])

for query, input, output in zip(queries, inputs, outputs):
    print("---------------------------------------------------------------------------------------------")
    input_token = tokenizer([araia_prompt.format(query,input,"")], return_tensors = "pt").to("cuda")

    with torch.no_grad():
        generated_tokens = model.generate(**input_token, max_new_tokens=256)
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    if "### Assistant:" in generated_text:
        assistant_response = generated_text.split("### Assistant:")[1].strip()
    else:
        assistant_response = generated_text.strip()

    idx += 1
    print(f"Finished sample {idx}: generation succesfull.")
    
    entry = {"reference": output, "llm": assistant_response}
    append_last_entry(output_file, [entry])

    #text_streamer = TextStreamer(tokenizer)
    #_ = model.generate(**input_token, streamer = text_streamer, max_new_tokens = 128)
