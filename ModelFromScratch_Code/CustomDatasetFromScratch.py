import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

class ProsodyDataset(Dataset):
    def __init__(self, json_file_paths, processor):
        self.processor = processor
        self.examples = []

        print("inside ProsodyDataset")

        for path in json_file_paths:
            with open(path, "r") as f:
                data = json.load(f)

            words = data["features"]["words"]
            word_duration = data["features"]["word_duration"]
            energies = data["features"]["energy"]
            prominences = data["features"]["prominence"]
            f0s = data["features"]["f0_parameterized"]
            pause_before = data["features"]["pause_before"]
            pause_after = data["features"]["pause_after"]
            duration = data["features"]["duration"]


            assert len(words) == len(word_duration) == len(energies) == len(prominences) == len(f0s), "Length mismatch in prosodic features"

            tokenized = processor.tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                add_special_tokens=False
            )
            token_ids = tokenized.input_ids[0]  # shape: (seq_len,)

            f0_len = 30  # or any fixed size
            dur_len = 30

            prosody_features = [
                [wd, e, p, pb, pa] +
                pad_or_truncate(f0, f0_len) +
                pad_or_truncate(d, dur_len)
                for wd, e, p, f0, pb, pa, d in zip(word_duration, energies, prominences, f0s, pause_before, pause_after, duration)
            ]


            # prosody_features = [
            #     [wd, e, p, *f0, pb, pa, *d] for wd, e, p, f0, pb, pa, d in zip(word_duration, energies, prominences, f0s, pause_before, pause_after, duration)
            # ]
            prosody_tensor = torch.tensor(prosody_features, dtype=torch.float)

            if prosody_tensor.size(0) != token_ids.size(0):
                min_len = min(prosody_tensor.size(0), token_ids.size(0))
                token_ids = token_ids[:min_len]
                prosody_tensor = prosody_tensor[:min_len]

            dummy_input_features = torch.zeros((80, 3000), dtype=torch.float)

            self.examples.append({
                "input_features": dummy_input_features,
                "labels": token_ids,
                "prosody": prosody_tensor
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def prosody_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_features = torch.stack([item['input_features'] for item in batch])
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    prosody = pad_sequence([item['prosody'] for item in batch], batch_first=True, padding_value=0.0)
    return {
        'input_features': input_features,
        'labels': labels,
        'prosody': prosody
    }

def pad_or_truncate(vec, target_len):
    if len(vec) > target_len:
        return vec[:target_len]
    else:
        return vec + [0.0] * (target_len - len(vec))


