from datasets import load_dataset
from transformers import WhisperConfig, WhisperProcessor, WhisperTokenizerFast
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

    model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location="cpu"))
    model.eval()
    return model, prosody_dim

# processor = WhisperProcessor.from_pretrained("trained_whisper_with_prosody")
# if processor.tokenizer.pad_token is None:
#     processor.tokenizer.pad_token = processor.tokenizer.eos_token

tokenizer = WhisperTokenizerFast(
    vocab_file="./whisper_tokenizer/vocab.json",
    merges_file="./whisper_tokenizer/merges.txt",
    add_prefix_space=True
)

model, prosody_dim = load_whisper_from_scratch("trained_whisper_with_prosody")
device = next(model.parameters()).device
model.to(device)

# Updated get_surprisal
def get_surprisal(sentence, model, tokenizer, prosody_dim):
    tokens = tokenizer(
        sentence.split(),
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    input_ids = tokens.input_ids.to(model.device)
    labels = input_ids.clone()
    seq_len = input_ids.size(1)

    dummy_prosody = torch.zeros((1, seq_len, prosody_dim)).to(model.device)
    positions = torch.arange(seq_len).unsqueeze(0).to(model.device)
    dummy_input_features = torch.zeros((1, 80, 3000)).to(model.device)

    with torch.no_grad():
        outputs = model(
            input_features=dummy_input_features,
            labels=labels,
            prosody=dummy_prosody,
            positions=positions
        )
    return outputs.loss.item()


# BLiMP Evaluation
def evaluate_blimp_task_custom(task_name):
    ds = load_dataset("nyu-mll/blimp", task_name)
    correct = 0
    total = 0

    for ex in ds['train']:
        good_loss = get_surprisal(ex["sentence_good"], model, tokenizer, prosody_dim)
        bad_loss = get_surprisal(ex["sentence_bad"], model, tokenizer, prosody_dim)
        if good_loss < bad_loss:
            correct += 1
        total += 1

    return correct / total

# Evaluate
subtasks = ["npi_present_1", "principle_A_case_1", "adjunct_island", "anaphor_number_agreement", "determiner_noun_agreement_1"]
results_sub = {}

for task in subtasks:
    acc = evaluate_blimp_task_custom(task)
    results_sub[task] = acc
    print(f"{task}: {acc:.2%}")
