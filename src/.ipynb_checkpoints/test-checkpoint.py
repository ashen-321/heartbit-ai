from transformers import TorchAoConfig, Gemma3ForConditionalGeneration, AutoProcessor
import torch

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = Gemma3ForConditionalGeneration.from_pretrained(
    "alfredcs/torchrun-medgemma-27b-grpo-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(
    "alfredcs/torchrun-medgemma-27b-grpo-merged",
    padding_side="left",
    use_fast=True
)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to("cuda")

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))