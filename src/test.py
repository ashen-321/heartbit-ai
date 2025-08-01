# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("alfredcs/torchrun-medgemma-27b-grpo-merged", use_fast=True)
model = AutoModelForImageTextToText.from_pretrained("alfredcs/torchrun-medgemma-27b-grpo-merged", device_map="auto")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
