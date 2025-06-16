
import torch
from torch.utils.data import Dataset
import json
import numpy as np

class ProsodyDataset(Dataset):
    def __init__(self, json_files, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer

        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            words = data["features"]["words"]
            word_duration = data["features"]["word_duration"]
            energies = data["features"]["energy"]
            prominences = data["features"]["prominence"]
            f0s = data["features"]["f0_parameterized"]
            pause_before = data["features"]["pause_before"]
            pause_after = data["features"]["pause_after"]
            durations = data["features"]["duration"]

            # Summarize duration features (shape: [N_words, 4])
            summarized_durations = [self.summarize_duration(d) for d in durations]

            # One prosody vector per word
            word_prosody = [
                [wd, e, p, pb, pa, *f0, *summ_d]
                for wd, e, p, pb, pa, f0, summ_d in zip(word_duration, energies, prominences, pause_before, pause_after, f0s, summarized_durations)
            ]  # Shape: [N_words, prosody_dim]

            # Tokenize with word-to-token alignment
            encoding = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
            input_ids = encoding.input_ids[0]  # (T,)
            word_ids = encoding.word_ids()     # list of length T (or None)

            # Align prosody per token
            aligned_prosody = torch.tensor([word_prosody[idx] for idx in word_ids], dtype=torch.float)  # (T, prosody_dim)

            position_ids = torch.arange(len(input_ids))

            self.samples.append({
                "input_ids": input_ids,
                "prosody": aligned_prosody,
                "positions": position_ids,
                "labels": input_ids.clone()
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def summarize_duration(self, d):
        arr = np.array(d)
        return [arr.mean(), arr.std(), arr.max(), arr.min()]