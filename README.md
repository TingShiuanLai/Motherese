# Motherese Hypothesis

This project investigates how prosodic features—such as pitch, rhythm, and stress—impact the grammatical learning abilities of neural language models. Inspired by the **Motherese Hypothesis**, we extend OpenAI’s Whisper architecture to incorporate prosody and evaluate its syntactic learning capacity across child-directed (CHILDES) and adult-directed (LibriSpeech) speech datasets. Our results offer insights into how suprasegmental cues influence grammar acquisition, particularly in long-distance syntactic dependencies.

---

## 📁 Project Structure

```
├── data/                             # Audio + .cha data from CHILDES & LibriSpeech (not included for privacy)
├── Pretrained_whisper/              # Whisper pretrained model fine-tuned with prosodic features
├── ModelFromScratch_Code/           # Model trained from scratch with prosody, no positional embedding
├── PositionalEmbeddingModelFromScratch_Code/  # Model from scratch with prosody + positional embedding
├── whisper_positional_embedding/    # Positional embedding + prosody with tuned hyperparameters
├── WithoutProsodyModel/             # Text-only model, trained from scratch (no prosody)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🧪 Datasets

This project uses:
- **CHILDES (Brent, Sachs, Snow)** for child-directed speech (~105 hrs total)
- **LibriSpeech-ASR (train-clean-100)** for adult-directed speech (100.59 hrs)

⚠️ These datasets are **not included** in the repository due to licensing and privacy restrictions. Place your extracted and aligned `.wav` and `.cha` files under the `data/` directory.

Each corpus (e.g., `brent/`, `sachs/`, `snow/`, `libri/`) should follow this internal structure:

```
data/
└── corpus_name/
    ├── converted_wav/      # Preprocessed .wav files
    ├── lab_phonemes/       # Phoneme-level alignment (.lab)
    ├── lab_words/          # Word-level alignment (.lab)
    └── json_features/      # Extracted prosodic features (JSON)
```

---

## 🧠 Models Overview

| Folder                            | Model Type                                        | Prosody | Positional Embedding | Tuned |
|----------------------------------|--------------------------------------------------|---------|------------------------|--------|
| `Pretrained_whisper/`            | Whisper pretrained + prosody                     | ✅      | ✅ (in Whisper)         | ❌     |
| `ModelFromScratch_Code/`         | From scratch (prosody only)                      | ✅      | ❌                     | ❌     |
| `PositionalEmbeddingModelFromScratch_Code/` | From scratch with pos. embed + prosody      | ✅      | ✅                     | ❌     |
| `whisper_positional_embedding/`  | From scratch + pos. embed + tuned                | ✅      | ✅                     | ✅     |
| `WithoutProsodyModel/`           | Text-only model from scratch                     | ❌      | ✅                     | ✅     |

All models were evaluated on the **BLiMP benchmark**, covering a broad range of grammatical phenomena.

---

## ⚙️ Installation

```bash
git clone https://github.com/TingShiuanLai/Motherese.git
cd Motherese
pip install -r requirements.txt
```

---

## 📊 Results Summary

- **Prosody consistently improves performance** on the most challenging syntactic tasks, such as:
  - **wh-movement**
  - **Negative Polarity Item (NPI) licensing**
  - **Scope ambiguity resolution**

- **LibriSpeech models** generally benefit more from prosody, especially on tasks with complex syntax:
  - E.g., *sentential_negation_npi_licensor_present* improved from **0.36 → 0.71 F1**
  - *superlative_quantifiers_2* rose from **0.41 → 0.73 F1**

- In contrast, **CHILDES models** show mixed effects:
  - Prosody slightly hurts performance on top-easy subtasks
  - But **dramatically helps** on hard ones like:
    - *wh_vs_that_with_gap* → **0.02 → 0.68 F1**
    - *wh_vs_that_with_gap_long_distance* → **0.04 → 0.70 F1**

- **Conclusion:** Prosody aids **disambiguation in complex structures**, but its effectiveness depends on data domain and task type.

---

## 🔬 Methodology Overview

1. **Data Processing**
   - Denoised CHILDES audio
   - Forced alignment via Montreal Forced Aligner (MFA)
   - Extracted prosodic features (e.g., energy, f0, prominence)

2. **Feature Engineering**
   - Parameterized f0 via DCT
   - Pause, duration, and prominence computation
   - Combined into per-token JSON files

3. **Model Architecture**
   - Modified Whisper decoder to:
     - Embed prosody features
     - Concatenate token + prosody embeddings
   - Trained all models **from scratch** (tokenizer reused)

4. **Training & Evaluation**
   - Used **cross-entropy loss**, Adam optimizer (lr = 3e-5)
   - Evaluated via **BLiMP suite** (binary sentence classification)
   - Metrics: **Accuracy, F1, Precision, Recall** (overall + per-subtask)

---

## 🛠 How to Run

1. Place audio + annotation files under `data/`
2. Choose one of the model folders and run the training script (e.g., `train.py`, `evaluate.py`)
3. Inspect metrics JSON or logs to analyze grammar performance

---

## 🚀 Future Work

We aim to:
- Scale up training with larger datasets for better generalization
- Apply **curriculum learning**: starting with child-directed (prosody-rich) speech, then gradually introducing adult-directed data to refine syntax
- Broaden evaluations to include downstream tasks beyond BLiMP

---

## 👥 Contributors

- **Ting-Shiuan Lai** – Data processing, hyperparameter tuning, evaluation, report writing
- **Devana Perupurayil** – Model architecture, cluster setup, evaluation, report writing

---

## 🔗 Resources

- GitHub: [github.com/TingShiuanLai/Motherese](https://github.com/TingShiuanLai/Motherese)
- HuggingFace Dataset: [motherese-prosody-data](https://huggingface.co/datasets/tingshiuanlai/motherese-prosody-data)

---

## 📄 License

This project is intended for **research and educational** use only.
