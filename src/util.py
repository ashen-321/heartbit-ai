import json
import os
import requests
import shutil


# Read AOSS config
def read_aoss_config(file_path, key):
    with open(file_path, 'r') as file:
        for line in file:
            key_value_pairs = line.strip().split(':')
            if key_value_pairs[0] == key:
                return key_value_pairs[1].lstrip()
    
    return None


# Wipe directory of all contents
def empty_directory(directory_path):
    if not os.path.exists(directory_path):
        return

    for item in os.scandir(directory_path):
        if item.is_file():
            os.remove(item.path)
        elif item.is_dir():
            shutil.rmtree(item.path)


# Get text from speech
def get_transcription(audio_filename):
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
