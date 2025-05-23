import os
import json
import numpy as np
from feature_extractors import ProsodyFeatureExtractor

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    return obj

def save_features_to_json(extractor, output_dir="output_json"):
    os.makedirs(output_dir, exist_ok=True)

    for sample in extractor.samples:
        filename = sample["filename"]
        output_path = os.path.join(output_dir, f"{filename}.json")

        # Ensure everything inside is serializable
        serializable_sample = make_json_serializable(sample)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_sample, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(extractor.samples)} JSON files to: {output_dir}")


if __name__ == "__main__":
    extractor = ProsodyFeatureExtractor(
        lab_root=r"C:\your\path\to\dev-clean\lab_words",
        wav_root=r"C:\your\path\to\dev-clean\converted_wav",
        phoneme_lab_root=r"C:\your\path\to\dev-clean\lab_phonemes",
        celex_path=None,
        extract_f0=True,
        extract_energy=True,
        extract_word_duration=True,
        extract_duration=True,
        extract_pause_before=True,
        extract_pause_after=True,
        extract_prominence=True,
        f0_stress_localizer="full_curve"
    )

    # Change this to where you want the JSON files saved
    save_features_to_json(
        extractor,
        output_dir=r"C:\your\path\to\dev-clean\json_features"
    )
