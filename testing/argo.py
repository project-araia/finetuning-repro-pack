import requests
import json
import os


def climrr_query(prompt, model="gpt4o"):

    # API endpoint to POST
    url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

    # Data to be sent as a POST in JSON format
    data = {
        "user": os.getenv("ARGO_USER"),
        "model": model,
        "system": "Below is a User query that describes a task or a question, paired with an Input along with its context. Write the Assitant's response that appropriately completes the request. If the Input is missing you should ignore it.",
        "prompt": [prompt],
        "stop": [],
        "temperature": 1.3,
        "top_p": 0.92,
    }

    # Convert the dict to JSON
    payload = json.dumps(data)

    # Add a header stating that the content type is JSON
    headers = {"Content-Type": "application/json"}

    # Send POST request
    response = requests.post(url, data=payload, headers=headers)

    return response.status_code, response.json()["response"]


def linguistic_variance(prompt):

    # API endpoint to POST
    url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

    # Data to be sent as a POST in JSON format
    data = {
        "user": os.getenv("ARGO_USER"),
        "model": "gpt4o",
        "system": "Rephrase the given prompt using natural, grammatically correct English. Introduce linguistic variance in style, tone, or word choice, while keeping the meaning identical.",
        "prompt": [prompt],
        "stop": [],
        "temperature": 1.3,
        "top_p": 0.92,
    }

    # Convert the dict to JSON
    payload = json.dumps(data)

    # Add a header stating that the content type is JSON
    headers = {"Content-Type": "application/json"}

    # Send POST request
    response = requests.post(url, data=payload, headers=headers)

    return response.status_code, response.json()["response"]
