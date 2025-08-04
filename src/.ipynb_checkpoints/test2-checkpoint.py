from transformers import AutoProcessor, AutoModelForImageTextToText, TextStreamer
from PIL import Image
import requests
import torch
import os

model_id = "google/gemma-3n-e4b-it" #"alfredcs/gemma-3N-finetune"#

messages = [
    # {
    #     "role": "system",
    #     "content": [{"type": "text", "text": "You are a helpful assistant."}]
    # },
    {
        "role": "user",
        "content": [
            # {"type": "image", "image": "https://img.wattpad.com/cover/296265693-256-k674393.jpg"},
            # {"type": "image", "image": "https://p.turbosquid.com/ts-thumb/R7/OZS3Pv/Uqsz7sMj/rattlesnake_rigged_c4d_00/jpg/1565713390/1920x1080/fit_q87/080f2cb9f6455db4f8bca2483bc3f04446b73a2a/rattlesnake_rigged_c4d_00.jpg"},
            {"type": "audio", "audio" : 'audio.mp3'},
            # {"type": "text", "text": "I got a bite by the animal as shown in the pictures, please explain what action I should take. After that, transcribe the audio."}
            {"type": "text", "text": "transcribe the audio."}
        ]
    }
]

path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../input-files')
)
os.chdir(path)

processor = AutoProcessor.from_pretrained(model_id, device_map="cuda:3")
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype="auto", device_map="cuda:3",
)

input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True, return_dict=True,
        return_tensors="pt",
)
input_ids = input_ids.to(model.device, dtype=model.dtype)

outputs = model.generate(**input_ids, max_new_tokens=256)

text = processor.batch_decode(
    outputs,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(text[0])
