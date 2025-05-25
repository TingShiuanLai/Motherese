import os
import json
import argparse
import pickle
import numpy as np

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    return obj

def save_features_to_json(extractor, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for sample in extractor.samples:
        filename = sample["filename"]
        output_path = os.path.join(output_dir, f"{filename}.json")
        serializable_sample = make_json_serializable(sample)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_sample, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(extractor.samples)} JSON files to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Export prosody features from a pickled extractor to JSON.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the .pkl file containing the ProsodyFeatureExtractor")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for JSON files")

    args = parser.parse_args()

    # Load the extractor
    with open(args.pkl, "rb") as f:
        extractor = pickle.load(f)

    # Export to JSON
    save_features_to_json(extractor, args.out_dir)

if __name__ == "__main__":
    main()

