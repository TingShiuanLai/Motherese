import os
import re
import soundfile as sf
import numpy as np
import string
import argparse

def clean_utterance(text):
    # 1. Remove repair markers [/] and replace glosses
    text = re.sub(r'\[\/\]', '', text)
    text = re.sub(r'\b(\w+)\s*\[:\s*([^\]]+)\]', r'\2', text)
    text = re.sub(r'\[[^]]*\]', '', text)

    # 2. Remove punctuation entirely (including ?, ., !, etc.)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Normalize spacing and case
    text = re.sub(r'\s+', ' ', text)

    return text.strip().upper()


# def segment_audio_by_utterance(base_dir, output_dir, max_duration=None):
#     wav_root = base_dir
#     # wav_root = os.path.join(base_dir, "converted_wav_denoised")
#     cha_root = base_dir

#     os.makedirs(output_dir, exist_ok=True)

#     for file in os.listdir(wav_root):
#         if not file.endswith(".wav"):
#             continue

#         file_id = os.path.splitext(file)[0]
#         wav_path = os.path.join(wav_root, file)
#         cha_path = os.path.join(cha_root, f"{file_id}.cha")

#         if not os.path.exists(cha_path):
#             print(f"❌ Missing .cha file for {file_id}")
#             continue

#         try:
#             audio, sr = sf.read(wav_path)
#         except Exception as e:
#             print(f"❌ Failed to read audio {file_id}: {e}")
#             continue

#         with open(cha_path, "r", encoding="utf-8") as f:
#             lines = f.readlines()

#         utt_index = 0
#         for line in lines:
#             if line.startswith("*") and "" in line:
#                 try:
#                     # Extract timing
#                     match = re.search(r'\u0015(\d+)_(\d+)\u0015', line)
#                     if not match:
#                         continue
#                     start_ms, end_ms = int(match[1]), int(match[2])

#                     # Skip very long segments if needed
#                     if max_duration and (end_ms - start_ms) > max_duration * 1000:
#                         continue

#                     start_sample = int((start_ms / 1000) * sr)
#                     end_sample = int((end_ms / 1000) * sr)

#                     if end_sample > len(audio):
#                         continue

#                     segment_audio = audio[start_sample:end_sample]
#                     text = line.split("")[0].split(":", 1)[1].strip()
#                     cleaned_text = clean_utterance(text)

#                     if not cleaned_text:
#                         continue

#                     new_id = f"{file_id}_{utt_index:04d}"
#                     out_wav = os.path.join(output_dir, f"{new_id}.wav")
#                     out_txt = os.path.join(output_dir, f"{new_id}.txt")

#                     sf.write(out_wav, segment_audio, sr)
#                     with open(out_txt, "w", encoding="utf-8") as out_f:
#                         out_f.write(cleaned_text + "\n")

#                     print(f"✔ Saved: {new_id}")
#                     utt_index += 1
#                 except Exception as e:
#                     print(f"⚠ Error in {file_id} utterance: {e}")
#                     continue

#     print("\n✅ Done segmenting all files.\n")

# # Example usage
# if __name__ == "__main__":
#     base_dir = r"C:\Users\10935\Desktop\Master\Spring 2025\DSC 291 Cognitive mod\final_project\Motherese\data\train\childes\Brent"
#     output_dir = os.path.join(base_dir, "utterance_level_clips")
#     segment_audio_by_utterance(base_dir, output_dir)
def segment_audio_by_utterance(base_dir, output_dir, wav_root=None, max_duration=None):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(base_dir):
        cha_files = [f for f in files if f.endswith(".cha")]

        for cha_file in cha_files:
            file_id = os.path.splitext(cha_file)[0]
            cha_path = os.path.join(root, cha_file)

            # Match corresponding .wav file
            wav_path = os.path.join(wav_root or root, os.path.relpath(root, base_dir), f"{file_id}.wav")
            if not os.path.exists(wav_path):
                print(f"❌ Missing .wav file for {file_id}")
                continue

            try:
                audio, sr = sf.read(wav_path)
            except Exception as e:
                print(f"❌ Failed to read audio {file_id}: {e}")
                continue

            with open(cha_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            utt_index = 0
            for line in lines:
                if line.startswith("*") and "" in line:
                    try:
                        match = re.search(r'\u0015(\d+)_(\d+)\u0015', line)
                        if not match:
                            continue
                        start_ms, end_ms = int(match[1]), int(match[2])

                        if max_duration and (end_ms - start_ms) > max_duration * 1000:
                            continue

                        start_sample = int((start_ms / 1000) * sr)
                        end_sample = int((end_ms / 1000) * sr)

                        if end_sample > len(audio) or end_sample <= start_sample:
                            continue

                        segment_audio = audio[start_sample:end_sample]
                        text = line.split("")[0].split(":", 1)[1].strip()
                        cleaned_text = clean_utterance(text)
                        if not cleaned_text:
                            continue

                        rel_path = os.path.relpath(root, base_dir)
                        save_dir = os.path.join(output_dir, rel_path)
                        os.makedirs(save_dir, exist_ok=True)

                        new_id = f"{file_id}_{utt_index:04d}"
                        out_wav = os.path.join(save_dir, f"{new_id}.wav")
                        out_txt = os.path.join(save_dir, f"{new_id}.txt")

                        sf.write(out_wav, segment_audio, sr)
                        with open(out_txt, "w", encoding="utf-8") as out_f:
                            out_f.write(cleaned_text + "\n")

                        print(f"✔ Saved: {new_id}")
                        utt_index += 1
                    except Exception as e:
                        print(f"⚠ Error in {file_id} utterance: {e}")
                        continue

    print("\n✅ Done segmenting all files.\n")

if __name__ == "__main__":
    base_dir = r"C:\Users\10935\Desktop\Master\Spring 2025\DSC 291 Cognitive mod\final_project\Motherese\data\train\childes\Snow"
    wav_root = r"C:\Users\10935\Desktop\Master\Spring 2025\DSC 291 Cognitive mod\final_project\Motherese\data\train\childes\Snow"
    output_dir = os.path.join(base_dir, "utterance_level_clips")

    segment_audio_by_utterance(base_dir, output_dir, wav_root=wav_root)