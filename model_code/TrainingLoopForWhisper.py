import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperTokenizerFast
from WhisperWrapper import WhisperWithProsody
from custom_dataset import ProsodyDataset  # assumes class is defined there

# Config
DATA_DIR = "/Users/devanaperupurayil/Documents/3rd_quarter/DSC_291/Final_project/model_code/training_data"
MODEL_NAME = "openai/whisper-small"
BATCH_SIZE = 1
EPOCHS = 30
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "trained_whisper_with_prosody"

print("After configurations")

# Load processor (for log-mel features)
tokenizer = WhisperTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)
# processor = WhisperProcessor(feature_extractor=processor.feature_extractor, tokenizer=tokenizer)
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer = tokenizer  # manually override with fast version

print("After loading processor")

# Collect all .json files
json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])

print("after collecting json files")

# Build dataset
dataset = ProsodyDataset([os.path.join(DATA_DIR, f) for f in json_files], processor)

print("after dataset")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("after building datasets")

# Load model
model = WhisperWithProsody(model_name=MODEL_NAME).to(DEVICE)

print("after loading the model")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("after optimizer step")

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_features = batch["input_features"].to(DEVICE)
        decoder_input_ids = batch["decoder_input_ids"].to(DEVICE)
        prosody_features = batch["prosody_features"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            prosody_features=prosody_features,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")
