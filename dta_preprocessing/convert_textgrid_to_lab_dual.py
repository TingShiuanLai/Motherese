from textgrid import TextGrid
import os
import glob

def textgrid_to_labs(textgrid_path, word_output_base, phoneme_output_base, relative_path):
    tg = TextGrid.fromFile(textgrid_path)

    # output file paths
    word_out = os.path.join(word_output_base, relative_path).replace(".TextGrid", ".lab")
    phoneme_out = os.path.join(phoneme_output_base, relative_path).replace(".TextGrid", ".lab")

    os.makedirs(os.path.dirname(word_out), exist_ok=True)
    os.makedirs(os.path.dirname(phoneme_out), exist_ok=True)

    # Extract word tier
    for tier in tg.tiers:
        if "word" in tier.name.lower():
            with open(word_out, "w", encoding="utf-8") as f:
                for interval in tier.intervals:
                    if interval.mark.strip():
                        f.write(f"{interval.minTime:.3f}\t{interval.maxTime:.3f}\t{interval.mark.strip()}\n")

        if "phone" in tier.name.lower() or "phoneme" in tier.name.lower():
            with open(phoneme_out, "w", encoding="utf-8") as f:
                for interval in tier.intervals:
                    if interval.mark.strip():
                        f.write(f"{interval.minTime:.3f}\t{interval.maxTime:.3f}\t{interval.mark.strip()}\n")


def batch_convert(base_dir, word_out_dir, phoneme_out_dir):
    textgrid_files = glob.glob(os.path.join(base_dir, "**", "*.TextGrid"), recursive=True)
    print(f"Found {len(textgrid_files)} TextGrid files.")

    for tg_path in textgrid_files:
        relative_path = os.path.relpath(tg_path, base_dir)
        textgrid_to_labs(tg_path, word_out_dir, phoneme_out_dir, relative_path)

    print("✅ Conversion complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Path to folder with TextGrid files")
    parser.add_argument("--word_out", type=str, required=True, help="Output folder for word-level .lab files")
    parser.add_argument("--phoneme_out", type=str, required=True, help="Output folder for phoneme-level .lab files")

    args = parser.parse_args()

    batch_convert(args.base_dir, args.word_out, args.phoneme_out)
