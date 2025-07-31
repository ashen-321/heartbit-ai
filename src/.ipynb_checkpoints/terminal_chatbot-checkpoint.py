from transformers import AutoModel, AutoTokenizer
import torch
import logging

# Logging
logger = logging.getLogger(__name__)

# Collect HF model and tokenizer
medgemma_grpo_id = "alfredcs/torchrun-medgemma-27b-grpo-merged"
medgemma_grpo = AutoModel.from_pretrained(medgemma_grpo_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(medgemma_grpo_id)
