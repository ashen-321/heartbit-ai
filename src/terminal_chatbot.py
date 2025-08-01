try:
    from transformers import AutoModel, AutoTokenizer
    import logging
    import os
except ImportError:
    raise ImportError('Error importing modules. Ensure all packages from ../requirements_local_chatbot.txt are installed. Run `pip '
          'install -r "requirements_local_chatbot.txt" in the terminal to install the packages.')
    

# Logging
logger = logging.getLogger(__name__)

# Model ID
medgemma_grpo_id = 'google/medgemma-4b-it' #"alfredcs/torchrun-medgemma-27b-grpo-merged"

# Model load path from disk
SAVED_MODEL_DIR = '../grpo-finetuned-model'
saved_model_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), SAVED_MODEL_DIR)
)

logging.info(f'Loading model {medgemma_grpo_id}...')

# Make directory if it doesn't exist
if not os.path.exists(saved_model_path):
    logging.debug(f'Directory at {saved_model_path} does not exist, creating new directory.')
    os.mkdir(saved_model_path)

try:
    # Collect HF model from local storage if available
    logging.debug(f'Getting model {medgemma_grpo_id} from local files.')
    medgemma_grpo = AutoModel.from_pretrained(saved_model_path)
    medgemma_tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
except OSError:
    # Collect HF model and tokenizer from HF if it's not stored locally
    logging.debug(f'Model {medgemma_grpo_id} not available locally, downloading from HuggingFace.')
    medgemma_grpo = AutoModel.from_pretrained(medgemma_grpo_id, device_map="auto")
    medgemma_tokenizer = AutoTokenizer.from_pretrained(medgemma_grpo_id)

    # Save model and tokenizer to disk
    medgemma_grpo.save_pretrained(saved_model_path)
