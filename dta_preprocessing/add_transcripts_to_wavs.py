import os

def add_transcripts(base_dir):
    wav_root = os.path.join(base_dir, "converted_wav")
    transcript_root = base_dir

    print(f"\nAdding transcripts next to WAV files...\n")

    for root, _, files in os.walk(wav_root):
        for file in files:
            if not file.endswith(".wav"):
                continue

            file_id = file.replace(".wav", "")  # e.g., 84-121123-0000
            rel_path = os.path.relpath(root, wav_root)
            try:
                speaker_id, chapter_id = rel_path.split(os.sep)
            except ValueError:
                print(f"Skipping malformed path: {rel_path}")
                continue

            # Original .trans.txt location
            trans_file = os.path.join(
                transcript_root, speaker_id, chapter_id, f"{speaker_id}-{chapter_id}.trans.txt"
            )

            if not os.path.exists(trans_file):
                print(f"❌ Transcript file missing: {trans_file}")
                continue

            # Load full .trans.txt and get the line for this file_id
            with open(trans_file, "r", encoding="utf-8") as f:
                transcripts = dict(
                    line.strip().split(" ", 1)
                    for line in f if " " in line
                )

            if file_id not in transcripts:
                print(f"❌ Transcript missing for {file_id} in {trans_file}")
                continue

            # Save single-line transcript next to .wav
            transcript_text = transcripts[file_id]
            transcript_txt_path = os.path.join(root, f"{file_id}.txt")
            with open(transcript_txt_path, "w", encoding="utf-8") as out_f:
                out_f.write(transcript_text + "\n")

            print(f"✔ {file_id} → {transcript_txt_path}")

    print("\n✅ Done adding transcript .txt files.\n")

# Example usage
if __name__ == "__main__":
    base_dir = "Data/Train/librispeech_asr/train-clean-100"  # change this if needed
    add_transcripts(base_dir)
