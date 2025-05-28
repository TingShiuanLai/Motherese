import os
import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperConfig
from CustomDatasetFromScratch import ProsodyDataset, prosody_collate_fn
from WhisperWrapperFromScratch import WhisperProsodyModel


# Configuration
DATA_DIR = "/Users/devanaperupurayil/Documents/3rd_quarter/DSC_291/Final_project/model_code/training_data"
BATCH_SIZE = 4
LEARNING_RATE = 3e-5
EPOCHS = 30
SAVE_DIR = "trained_whisper_with_prosody"

# Load and sort JSON files
json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
json_paths = [os.path.join(DATA_DIR, f) for f in json_files]

# Initialize tokenizer + feature extractor manually (not from pretrained)
tokenizer = WhisperTokenizer(
    vocab_file="./whisper_tokenizer/vocab.json",
    merges_file="./whisper_tokenizer/merges.txt"
)
feature_extractor = WhisperFeatureExtractor(sampling_rate=16000)
processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

# Dataset & Dataloader
dataset = ProsodyDataset(json_paths, processor)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=prosody_collate_fn)

# Create new config and initialize model from scratch
prosody_dim = 65
config = WhisperConfig()
model = WhisperProsodyModel(config=config, prosody_dim=prosody_dim)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in dataloader:
        input_features = batch['input_features']
        labels = batch['labels']
        prosody = batch['prosody']

        outputs = model(input_features=input_features, labels=labels, prosody=prosody)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save model
model.save_pretrained(SAVE_DIR)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pytorch_model.bin"))
processor.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")

