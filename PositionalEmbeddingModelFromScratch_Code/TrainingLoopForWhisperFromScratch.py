
import os
import torch
from torch.utils.data import DataLoader
from transformers import WhisperConfig, WhisperTokenizerFast
from WhisperWrapperFromScratch import WhisperProsodyModel
from CustomDatasetFromScratch import ProsodyDataset
import json

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

# Set up
DATA_DIR = "/Users/devanaperupurayil/Documents/3rd_quarter/DSC_291/Final_project/ProsodyEmbeddingModel/training_data"
BATCH_SIZE = 4
LEARNING_RATE = 3e-5
EPOCHS = 30
SAVE_DIR = "trained_whisper_with_prosody"

# Load and sort JSON files
json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
json_paths = [os.path.join(DATA_DIR, f) for f in json_files]

# Initialize tokenizer + feature extractor manually (not from pretrained)
tokenizer = WhisperTokenizerFast(
    vocab_file="./whisper_tokenizer/vocab.json",
    merges_file="./whisper_tokenizer/merges.txt",
    add_prefix_space=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = ProsodyDataset(json_paths, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

print("dataLoader done")

# Create model from scratch
prosody_dim = 13  # update if needed, depending on the dimensions of the features
config = WhisperConfig()
model = WhisperProsodyModel(config=config, prosody_dim=prosody_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.train()


# # Training loop
# for epoch in range(EPOCHS):
#     for batch in dataloader:
#         outputs = model(
#             input_features=torch.zeros(batch["labels"].shape[0], 80, 3000).to(model.device),
#             prosody=batch["prosody"],
#             positions=batch["positions"],
#             labels=batch["labels"]
#             # attention_mask=batch["attention_mask"]
#         )
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in dataloader:
        input_features = torch.zeros(batch["labels"].shape[0], 80, 3000).to(model.device)
        labels = batch['labels']
        prosody = batch['prosody']
        positions=batch["positions"]

        outputs = model(input_features=input_features, labels=labels, prosody=prosody, positions=positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save model
model.save_pretrained(SAVE_DIR)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))

# processor.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")
