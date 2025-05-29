from transformers import WhisperTokenizer

# Load the tokenizer from the Hugging Face hub
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

# Save the tokenizer files locally
tokenizer.save_pretrained("./whisper_tokenizer")
