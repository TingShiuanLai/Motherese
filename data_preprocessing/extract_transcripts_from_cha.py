# import os
# import re

# def clean_utterance(text):
#     # Remove repair markers, colons, square brackets, etc.
#     text = re.sub(r'\[.*?\]', '', text)          # remove bracketed annotations
#     text = re.sub(r'[:/]', '', text)             # remove ':' and '/' markers
#     text = re.sub(r'\s+', ' ', text)             # normalize whitespace
#     return text.strip().upper()

# def extract_transcripts_plain(base_dir):
#     wav_root = os.path.join(base_dir, "converted_wav_denoised")
#     cha_root = base_dir

#     print(f"\nExtracting plaintext transcripts from .cha files...\n")

#     for root, _, files in os.walk(wav_root):
#         for file in files:
#             if not file.endswith(".wav"):
#                 continue

#             file_id = os.path.splitext(file)[0]
#             cha_path = os.path.join(cha_root, f"{file_id}.cha")
#             output_txt_path = os.path.join(root, f"{file_id}.txt")

#             if not os.path.exists(cha_path):
#                 print(f"❌ Missing .cha file: {cha_path}")
#                 continue

#             with open(cha_path, "r", encoding="utf-8") as f:
#                 lines = f.readlines()

#             utterances = []
#             for line in lines:
#                 if line.startswith("*") and " " in line:
#                     try:
#                         text = line.split(" ")[0].split(":", 1)[1].strip()
#                         cleaned = clean_utterance(text)
#                         if cleaned:
#                             utterances.append(cleaned)
#                     except IndexError:
#                         continue

#             if utterances:
#                 with open(output_txt_path, "w", encoding="utf-8") as out_f:
#                     out_f.write("\n".join(utterances) + "\n")
#                 print(f"✔ Created: {output_txt_path}")
#             else:
#                 print(f"⚠ No valid utterances found in {cha_path}")

#     print("\n✅ Done extracting transcript .txt files.\n")

# # Example usage
# if __name__ == "__main__":
#     base_dir = r"C:\Users\10935\Desktop\Master\Spring 2025\DSC 291 Cognitive mod\final_project\Motherese\data\train\childes\Sachs"
#     extract_transcripts_plain(base_dir)
import os
import re
import soundfile as sf

def clean_utterance(text):
    text = re.sub(r'\[.*?\]', '', text)          # remove bracketed annotations
    text = re.sub(r'[:/]', '', text)             # remove ':' and '/' markers
    text = re.sub(r'\s+', ' ', text)             # normalize whitespace
    return text.strip().upper()

def extract_transcripts_plain(base_dir):
    wav_root = os.path.join(base_dir, "converted_wav_denoised_truncated")  # new folder
    cha_root = base_dir

    print(f"\nExtracting transcripts matching truncated audio...\n")

    for root, _, files in os.walk(wav_root):
        for file in files:
            if not file.endswith(".wav"):
                continue

            file_id = os.path.splitext(file)[0]
            wav_path = os.path.join(root, file)
            cha_path = os.path.join(cha_root, f"{file_id}.cha")
            output_txt_path = os.path.join(root, f"{file_id}.txt")

            if not os.path.exists(cha_path):
                print(f"❌ Missing .cha file: {cha_path}")
                continue

            try:
                audio, sr = sf.read(wav_path)
                max_time_ms = int((len(audio) / sr) * 1000)
            except Exception as e:
                print(f"❌ Failed to read {wav_path}: {e}")
                continue

            with open(cha_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            utterances = []
            for line in lines:
                if line.startswith("*") and "" in line:
                    try:
                        # extract timing
                        time_match = re.findall(r'\u0015(\d+)_(\d+)\u0015', line)
                        if not time_match:
                            continue
                        start, end = map(int, time_match[0])
                        if end > max_time_ms:
                            continue  # skip if beyond truncated audio

                        # extract and clean text
                        text = line.split("")[0].split(":", 1)[1].strip()
                        cleaned = clean_utterance(text)
                        if cleaned:
                            utterances.append(cleaned)
                    except Exception:
                        continue

            if utterances:
                with open(output_txt_path, "w", encoding="utf-8") as out_f:
                    out_f.write("\n".join(utterances) + "\n")
                print(f"✔ Created: {output_txt_path}")
            else:
                print(f"⚠ No valid utterances within audio duration for {file_id}")

    print("\n✅ Done extracting transcripts from truncated audio.\n")

# Example usage
if __name__ == "__main__":
    base_dir = r"C:\Users\10935\Desktop\Master\Spring 2025\DSC 291 Cognitive mod\final_project\Motherese\data\train\childes\Sachs"
    extract_transcripts_plain(base_dir)
