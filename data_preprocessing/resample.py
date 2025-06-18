import os
import soundfile as sf
import shutil
import random
from tqdm import tqdm

def get_wav_duration(filepath):
    with sf.SoundFile(filepath) as f:
        return len(f) / f.samplerate

def find_complete_trio(wav_root):
    """Find .wav files with matching .txt and .TextGrid files"""
    trio_pairs = []
    print("🔍 Scanning for .wav + .txt + .TextGrid trios...")
    for root, _, files in tqdm(list(os.walk(wav_root))):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                base_path = os.path.splitext(wav_path)[0]
                txt_path = base_path + '.txt'
                tg_path = base_path + '.TextGrid'
                if os.path.exists(txt_path) and os.path.exists(tg_path):
                    trio_pairs.append((wav_path, txt_path, tg_path))
    return trio_pairs

def select_files_for_duration(wav_txt_tg_trios, target_seconds):
    random.shuffle(wav_txt_tg_trios)
    selected = []
    total_duration = 0.0

    print("🎯 Selecting files until reaching target duration...")
    for wav_path, txt_path, tg_path in tqdm(wav_txt_tg_trios):
        dur = get_wav_duration(wav_path)
        if total_duration + dur <= target_seconds:
            selected.append((wav_path, txt_path, tg_path))
            total_duration += dur
        if total_duration >= target_seconds:
            break

    return selected, total_duration

def copy_selected_trios(trios, input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    print("📦 Copying selected files...")
    for wav_path, txt_path, tg_path in tqdm(trios):
        for path in [wav_path, txt_path, tg_path]:
            rel_path = os.path.relpath(path, input_root)
            dst_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(path, dst_path)

# --- SETTINGS ---
input_folder = r"C:\Users\10935\Desktop\Master\Spring 2025\DSC 291 Cognitive mod\final_project\Motherese\data\train\childes\Brent\utterance_level_clips"
output_folder = r"C:\Users\10935\Desktop\Master\Spring 2025\DSC 291 Cognitive mod\final_project\Motherese\data\train\childes\Brent_resample\utterance_level_clips"
target_duration_secs = 5.50 * 3600  # 5.5 hours = 19800 seconds

# --- EXECUTION ---
all_trios = find_complete_trio(input_folder)
selected_trios, total_dur = select_files_for_duration(all_trios, target_duration_secs)
copy_selected_trios(selected_trios, input_folder, output_folder)

print(f"\n✅ Copied {len(selected_trios)} .wav+.txt+.TextGrid trios to '{output_folder}'")
print(f"⏱️ Total duration: {total_dur / 3600:.2f} hours")
