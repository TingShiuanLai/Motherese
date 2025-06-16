import warnings
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated")

import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau  # or other scheduler like CosineAnnealingLR
from transformers import WhisperConfig, WhisperTokenizerFast
from WhisperWrapperFromScratch import WhisperProsodyModel
from CustomDatasetFromScratch import ProsodyDataset
import json
import yaml
import time
import matplotlib.pyplot as plt


config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    input_ids = [item["input_ids"] for item in batch]
    prosody = [item["prosody"] for item in batch]
    positions = [item["positions"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    prosody_padded = pad_sequence(prosody, batch_first=True, padding_value=0.0)
    positions_padded = pad_sequence(positions, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 for ignore_index

    attention_mask = (input_ids_padded != 0).long()

    return {
        "input_ids": input_ids_padded,
        "prosody": prosody_padded,
        "positions": positions_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask
    }

# # Set up
# DATA_DIR = "/Users/devanaperupurayil/Documents/3rd_quarter/DSC_291/Final_project/ProsodyEmbeddingModel/training_data"
# BATCH_SIZE = 4
# LEARNING_RATE = 3e-5
# EPOCHS = 30
# SAVE_DIR = "trained_whisper_with_prosody"
DATA_DIR = cfg["data_dir"]
SAVE_DIR = cfg["save_dir"]
LEARNING_RATE = float(cfg["learning_rate"])
BATCH_SIZE = int(cfg["batch_size"])
EPOCHS = int(cfg["epochs"])
prosody_dim = int(cfg["prosody_dim"])



# Load and sort JSON files
json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
json_paths = [os.path.join(DATA_DIR, f) for f in json_files]

# Initialize tokenizer + feature extractor manually (not from pretrained)
# tokenizer = WhisperTokenizerFast(
#     vocab_file="./whisper_tokenizer/vocab.json",
#     merges_file="./whisper_tokenizer/merges.txt",
#     add_prefix_space=True
# )
tokenizer = WhisperTokenizerFast(
    vocab_file=cfg["vocab_path"],
    merges_file=cfg["merges_path"],
    add_prefix_space=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = ProsodyDataset(json_paths, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

print("dataLoader done")

# Create model from scratch
# prosody_dim = 13  # update if needed, depending on the dimensions of the features
config = WhisperConfig()
model = WhisperProsodyModel(config=config, prosody_dim=prosody_dim) # verbose=True

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # verbose=True

history = {
    "epoch": [],
    "loss": [],
    "accuracy": [],
    "lr": [],
    "time": []
}

for epoch in range(EPOCHS):
    start_time = time.time()
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0

    model.train()
    for batch in dataloader:
        input_features = torch.zeros(batch["labels"].shape[0], 80, 3000).to(model.device)
        labels = batch['labels'].to(model.device)
        prosody = batch['prosody'].to(model.device)
        positions = batch["positions"].to(model.device)

        outputs = model(input_features=input_features, labels=labels, prosody=prosody, positions=positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct_tokens += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - start_time

    # Record for plotting
    history["epoch"].append(epoch + 1)
    history["loss"].append(avg_loss)
    history["accuracy"].append(accuracy)
    history["lr"].append(lr)
    history["time"].append(epoch_time)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}, Time: {epoch_time:.2f}s, LR: {lr}")
    scheduler.step(avg_loss)

# for epoch in range(EPOCHS):
#     total_loss = 0.0
#     for batch in dataloader:
#         input_features = torch.zeros(batch["labels"].shape[0], 80, 3000).to(model.device)
#         labels = batch['labels']
#         prosody = batch['prosody']
#         positions=batch["positions"]

#         outputs = model(input_features=input_features, labels=labels, prosody=prosody, positions=positions)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(dataloader)
#     print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

#     scheduler.step(avg_loss)  # 🔁 dynamically adjust LR

# Save model
model.save_pretrained(SAVE_DIR)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))

# Save training history
import pandas as pd
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)

# Plot loss curve
plt.figure(figsize=(6, 4))
plt.plot(history["epoch"], history["loss"], marker='o', label="Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
plt.show()

# Total training time
total_time = sum(history["time"])
print(f"\nTotal training time: {total_time:.2f} seconds")
print(f"Model and history saved to {SAVE_DIR}")