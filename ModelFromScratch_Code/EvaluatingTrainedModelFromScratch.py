from datasets import load_dataset
from transformers import WhisperConfig, WhisperProcessor
from WhisperWrapperFromScratch import WhisperProsodyModel
import torch
import json
import os

# Load model and processor

def load_whisper_from_scratch(save_directory):
    with open(os.path.join(save_directory, "prosody_config.json")) as f:
        prosody_dim = json.load(f)["prosody_dim"]
    config = WhisperConfig.from_pretrained(save_directory)
    model = WhisperProsodyModel(config=config, prosody_dim=prosody_dim)

    # Load the saved weights (ensure they include the prosody_projection)
    model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location="cpu"))

    return model

processor = WhisperProcessor.from_pretrained("trained_whisper_with_prosody")

# Manually set pad token to eos token if missing
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = load_whisper_from_scratch("trained_whisper_with_prosody")
device = next(model.parameters()).device

def get_surprisal(sentence, model, processor, prosody_dim=65):
    inputs = processor.tokenizer(sentence, return_tensors="pt", padding=True)
    decoder_input_ids = inputs.input_ids

    # Dummy spectrograms and dummy prosody features
    batch_size, seq_len = decoder_input_ids.shape
    dummy_input_features = torch.zeros((batch_size, 80, 3000))  # (B, features, timesteps)
    # dummy_input_features = dummy_input_features.transpose(1, 2)  # to (B, T, 80) as expected

    dummy_prosody = torch.zeros((batch_size, seq_len, prosody_dim))

    with torch.no_grad():
        outputs = model(
            input_features=dummy_input_features,
            labels=decoder_input_ids,
            prosody=dummy_prosody
        )
        return outputs.loss.item()


def evaluate_blimp_task_custom(task_name):
    ds = load_dataset("nyu-mll/blimp", task_name)
    correct = 0
    total = 0

    for ex in ds['train']:
        good_loss = get_surprisal(ex["sentence_good"], model, processor)
        bad_loss = get_surprisal(ex["sentence_bad"], model, processor)
        if good_loss < bad_loss:
            correct += 1
        total += 1

    return correct / total

# Evaluate selected subtasks
subtasks = ["npi_present_1", "principle_A_case_1", "adjunct_island", "anaphor_number_agreement", "determiner_noun_agreement_1"]
results_sub = {}

for task in subtasks:
    acc = evaluate_blimp_task_custom(task)
    results_sub[task] = acc
    print(f"{task}: {acc:.2%}")
