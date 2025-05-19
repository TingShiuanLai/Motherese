import os
import argparse
import soundfile as sf

def convert_flac_to_wav(base_dir):
    input_dir = base_dir
    output_dir = os.path.join(base_dir, "converted_wav")

    print(f"\nConverting .flac → .wav\nFrom: {input_dir}\nTo:   {output_dir}\n")

    # Create the output root folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                flac_path = os.path.join(root, file)
                audio_data, sr = sf.read(flac_path)

                # Reconstruct relative path and target
                rel_path = os.path.relpath(root, input_dir)
                target_dir = os.path.join(output_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)

                wav_path = os.path.join(target_dir, file.replace(".flac", ".wav"))
                sf.write(wav_path, audio_data, sr)

                print(f"✔ Converted: {flac_path} → {wav_path}")
                count += 1

    print(f"\n✅ Done. {count} files converted.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all FLAC files to WAV under a base directory")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing speaker/chapter folders with .flac files (e.g., Data/Train/librispeech_asr/dev-clean)")
    args = parser.parse_args()
    convert_flac_to_wav(args.base_dir)
