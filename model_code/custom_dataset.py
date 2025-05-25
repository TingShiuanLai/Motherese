from torch.utils.data import Dataset
from transformers import WhisperTokenizer
import torch
import json

class ProsodyDataset(Dataset):
    def __init__(self, json_paths, tokenizer: WhisperTokenizer):
        print("inside ProsodyDataset")
        self.tokenizer = tokenizer
        self.data = []

        for path in json_paths:
            with open(path, "r") as f:
                ex = json.load(f)

            words = ex["features"]["words"]
            text = ex["text"].lower()
            duration = ex["features"]["word_duration"]
            energy = ex["features"]["energy"]
            prominence = ex["features"]["prominence"]
            f0 = ex["features"]["f0_parameterized"]

            # Tokenize each word (ensure 1:1 mapping)
            tokenized = tokenizer.tokenizer(words, is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
            token_ids = tokenized.input_ids[0]  # shape (T,)

            # Prepare prosody features
            prosody = torch.tensor([
                [d, e, p, *f] for d, e, p, f in zip(duration, energy, prominence, f0)
            ], dtype=torch.float)  # shape (T, 7)

            # Align prosody features to tokenization
            # Get alignment from tokenizer
            word_ids = tokenized.word_ids()  # list of word indices per token

            # Expand prosody features to match tokens
            expanded_prosody = []
            for i, word_id in enumerate(word_ids):
                if word_id is None:
                    continue  # skip special tokens if any (though unlikely since add_special_tokens=False)
                expanded_prosody.append(prosody[word_id])

            prosody = torch.stack(expanded_prosody, dim=0)


            # Use dummy encoder input (shape = (80, 3000) is standard log-mel shape)
            input_features = torch.randn(80, 3000)

            # Create decoder labels (shifted tokens)
            labels = token_ids.clone()
            labels[labels == tokenizer.tokenizer.pad_token_id] = -100  # mask pad tokens

            self.data.append({
                "input_features": input_features,
                "decoder_input_ids": token_ids,
                "prosody_features": prosody,
                "labels": labels,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
