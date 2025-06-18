import sys
import os

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
import argparse
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

        serializable_sample = make_json_serializable(sample)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_sample, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(extractor.samples)} JSON files to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prosody features and save as JSON and Pickle")
    parser.add_argument("--lab_root", required=True, help="Path to lab_words directory")
    parser.add_argument("--wav_root", required=True, help="Path to converted_wav directory")
    parser.add_argument("--phoneme_lab_root", required=True, help="Path to lab_phonemes directory")
    parser.add_argument("--output_json", required=True, help="Directory to save JSON files")
    parser.add_argument("--pkl_path", required=True, help="File path to save the extractor as a .pkl file")

    args = parser.parse_args()

    extractor = ProsodyFeatureExtractor(
        lab_root=args.lab_root,
        wav_root=args.wav_root,
        phoneme_lab_root=args.phoneme_lab_root,
        celex_path=r"C:\Users\10935\Desktop\Master\Spring 2025\DSC 291 Cognitive mod\final_project\Motherese\celex2\english\epw\epw.cd",
        extract_f0=True,
        extract_energy=True,
        extract_word_duration=True,
        extract_duration=True,
        extract_pause_before=True,
        extract_pause_after=True,
        extract_prominence=True,
        f0_stress_localizer="celex",
        f0_mode="dct",
        f0_n_coeffs=4,
        energy_mode="mean",
        word_duration_mode="syllable_norm",
        prominence_mode="mean"
    )

    with open(args.pkl_path, "wb") as f:
        pickle.dump(extractor, f)

    print(f"✅ Extractor computed and saved to: {args.pkl_path}")
    
    save_features_to_json(extractor, output_dir=args.output_json)
    
    print(f"✅ Prosody features saved to JSON files in: {args.output_json}")