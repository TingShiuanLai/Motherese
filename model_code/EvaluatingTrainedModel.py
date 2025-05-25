from datasets import load_dataset
from transformers import WhisperProcessor
from WhisperWrapper import WhisperWithProsody
import torch
import json
import os

# Load model and processor

def load_whisper_with_prosody(save_directory):
    with open(os.path.join(save_directory, "prosody_config.json")) as f:
        config = json.load(f)
    model = WhisperWithProsody(model_name=save_directory, prosody_dim=config["prosody_dim"])
    return model

processor = WhisperProcessor.from_pretrained("trained_whisper_with_prosody")
# model = WhisperForConditionalGenerationWithProsody.from_pretrained("path/to/your/trained/model")
# model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
model = load_whisper_with_prosody("trained_whisper_with_prosody")
device = next(model.parameters()).device

def get_surprisal(sentence, model, processor, prosody_dim=7):
    inputs = processor.tokenizer(sentence, return_tensors="pt", padding=True)
    decoder_input_ids = inputs.input_ids.to(device)

    # Dummy prosody: (batch_size, seq_len, prosody_dim)
    seq_len = decoder_input_ids.shape[1]
    dummy_prosody = torch.zeros((1, seq_len, prosody_dim)).to(device)

    with torch.no_grad():
        outputs = model(
            input_features=None,  # since we’re not using audio
            decoder_input_ids=decoder_input_ids,
            prosody_features=dummy_prosody,
            labels=decoder_input_ids,  # compute loss on full sequence
        )
        return outputs["loss"].item()

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
subtasks = ["npi_present_1", "principle_A_case_1"]
results_sub = {}

for task in subtasks:
    acc = evaluate_blimp_task_custom(task)
    results_sub[task] = acc
    print(f"{task}: {acc:.2%}")
