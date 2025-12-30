import os
import climparser
import templater
import argo
import json
import requests
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager
from alive_progress import alive_bar

MODEL = "gemini25pro"

# --- Load chat templates with placeholder-based questions and answers ---
#chat_templates = templater.load_template(f'{os.getenv("PROJECT_HOME")}/datasets/ClimRR_Dataset_Test_new_n_final_n.json')
chat_templates = templater.load_template(f'{os.getenv("PROJECT_HOME")}/datasets/ClimRR_Dataset_Test_filtered_new_n_final_n.json')
evaluation_entries = Manager().list()


def evaluate_template(template, progress_bar):
    reference_response = template["assistant"]
    template["assistant"] = ""

    status_code, llm_response = argo.climrr_query(json.dumps(template), MODEL)
    evaluation_entries.append({"reference": reference_response, "llm": llm_response})

    progress_bar()


# Parallel processing with tqdm for progress tracking
with alive_bar(total=len(chat_templates), title="Evaluating Templates") as progress_bar:
    Parallel(n_jobs=10, backend="threading")(
        delayed(evaluate_template)(template, progress_bar)
        for template in chat_templates
    )

templater.append_entries(f'{os.getenv("PROJECT_HOME")}/evaluations/Evaluation_{MODEL.upper()}.json', list(evaluation_entries))
