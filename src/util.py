import json
import os
import requests
import shutil
from openai import OpenAI
from openai._types import NotGiven, NOT_GIVEN


def empty_directory(directory_path):
    if not os.path.exists(directory_path):
        return

    # Wipe directory of all contents
    for item in os.scandir(directory_path):
        if item.is_file():
            os.remove(item.path)
        elif item.is_dir():
            shutil.rmtree(item.path)


def get_asr(audio_filename):
    file_size = os.path.getsize(audio_filename)
    if file_size == 0:
        return 'No audio.'

    # Set the API endpoint
    url = 'http://video.cavatar.info:8082/generate'

    # Define HTTP headers
    headers = {
        'Accept': 'application/json',
    }

    # Define the file to be uploaded
    files = {
        'audio_file': (audio_filename, open(audio_filename, 'rb'), 'audio/mpeg')
    }

    # Make the POST request
    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        output = json.dumps(response.json(), indent=3).replace('"', '')
        return output
    else:
        return ""


def openai_generate(model_id: str, messages, max_tokens: int, tools: list | NotGiven = NOT_GIVEN):
    openai_api_key = "EMPTY"
    openai_api_base = "http://mcp1.cavatar.info:8081/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant. Please answer the user question accurately and truthfully."},
            {"role": "user", "content": messages},
        ],
        max_tokens=max_tokens,
        tools=tools
    )
    return chat_response
