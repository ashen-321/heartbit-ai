try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import logging
    import os
    import asyncio
    import torch
except ImportError:
    raise ImportError('Error importing modules. Ensure all packages from ../requirements_local_chatbot.txt are installed. Run `pip '
          'install -r requirements_local_chatbot.txt` in the terminal to install the packages.')
    

# Logging
logger = logging.getLogger(__name__)

# Create ../input-files if it does not exist
input_relative_path = '../input-files'
input_abspath = os.path.normpath(
    os.path.join(os.path.dirname(__file__), input_relative_path)
)
if not os.path.exists(input_abspath):
    logger.debug(f'Input files directory at {input_abspath} does not exist, creating new directory.')
    os.mkdir(input_abspath)


# Model ID
medgemma_grpo_id = "google/gemma-3n-e4b-it" #"alfredcs/gemma-3N-finetune"

# Collect model and tokenizer from HF
logger.debug(f'Loading model {medgemma_grpo_id} from HuggingFace...')

device_map = "cuda:0" if torch.cuda.is_available() else "cpu"  # Using multiple GPUs causes issues
processor = AutoProcessor.from_pretrained(medgemma_grpo_id, device_map=device_map)
model = AutoModelForImageTextToText.from_pretrained(
    medgemma_grpo_id, torch_dtype="auto", device_map=device_map,
)


# See if file is an image or audio based on the extension
def get_file_type(extension: str) -> str:
    match extension:
        case ".jpg" | ".png" | ".webp":
            return "image"
        case ".mp3" | ".wav":
            return "audio"
        case _:
            return "UNKNOWN"


# Process a query
async def process_query(query: str):
    global messages

    # Add relevant files to message content
    message_content = []
    for entry in os.listdir(input_abspath):
        entry_abspath = os.path.join(input_abspath, entry)

        # Continue if entry is a directory
        if not os.path.isfile(entry_abspath):
            continue

        # Continue if file is marked with a #
        if entry.startswith("#"):
            continue

        # Add message content based on image type
        _, file_extension = os.path.splitext(entry_abspath)
        file_type = get_file_type(file_extension)
        file_relpath = os.path.join(input_relative_path, entry)
        message_content.append({"type": file_type, file_type: file_relpath})

    # Add query to message content
    message_content.append({"type": "text", "text": query})

    # Add all content to message memory
    messages.append({"role": "user", "content": message_content})

    # Call model with messages
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
    )[0]

    # Save model output to message memory
    messages.append({"role": "assistant", "content": text})
    return text


# Main chat loop
messages = []
while True:
    try:
        query = input("\nQuery: ").strip()

        # Exit the program if the user enters "quit"
        if query.lower() == "quit":
            break

        # Wipe all message memory if the user enters "wipe"
        if query.lower() == "wipe":
            messages = []
            continue

        response = asyncio.run(process_query(query))
        print("\n" + response)

    except Exception as e:
        print(f"\n{type(e).__name__}: {str(e)}")processor = AutoProcessor.from_pretrained(medgemma_grpo_id, device_map=device_map)
