# import os
# import argparse
# import soundfile as sf
# from pydub import AudioSegment

# def convert_audio_to_wav(base_dir, input_format):
#     input_dir = base_dir
#     output_dir = os.path.join(base_dir, "converted_wav")

#     print(f"\nConverting .{input_format} → .wav\nFrom: {input_dir}\nTo:   {output_dir}\n")

#     os.makedirs(output_dir, exist_ok=True)
#     count = 0

#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith(f".{input_format}"):
#                 input_path = os.path.join(root, file)

#                 rel_path = os.path.relpath(root, input_dir)
#                 target_dir = os.path.join(output_dir, rel_path)
#                 os.makedirs(target_dir, exist_ok=True)

#                 wav_path = os.path.join(target_dir, file.replace(f".{input_format}", ".wav"))

#                 try:
#                     if input_format == "flac":
#                         audio_data, sr = sf.read(input_path)
#                         sf.write(wav_path, audio_data, sr)
#                     elif input_format == "mp3":
#                         audio = AudioSegment.from_mp3(input_path)
#                         audio.export(wav_path, format="wav")
#                     else:
#                         print(f"❌ Unsupported format: {input_format}")
#                         return

#                     print(f"✔ Converted: {input_path} → {wav_path}")
#                     count += 1

#                 except Exception as e:
#                     print(f"❌ Failed to convert {input_path}: {e}")

#     print(f"\n✅ Done. {count} files converted.\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert audio files to WAV under a base directory")
#     parser.add_argument("--base_dir", type=str, required=True,
#                         help="Base directory containing folders with audio files")
#     parser.add_argument("--input_format", type=str, choices=["flac", "mp3"], required=True,
#                         help="Input audio format to convert (flac or mp3)")
#     args = parser.parse_args()
#     convert_audio_to_wav(args.base_dir, args.input_format)
import os
import argparse
import soundfile as sf
from pydub import AudioSegment
import librosa
import noisereduce as nr
import subprocess

def convert_audio_to_wav(base_dir, input_format, denoise=False):
    input_dir = base_dir
    output_dir = os.path.join(base_dir, "converted_wav_denoised" if denoise else "converted_wav")

    print(f"\nConverting .{input_format} → .wav {'with noise reduction' if denoise else ''}")
    print(f"From: {input_dir}\nTo:   {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(f".{input_format}"):
                input_path = os.path.join(root, file)

                rel_path = os.path.relpath(root, input_dir)
                target_dir = os.path.join(output_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)

                wav_path = os.path.join(target_dir, file.replace(f".{input_format}", ".wav"))
                temp_wav_path = wav_path.replace(".wav", "_temp.wav")

                try:
                    if input_format == "flac":
                        audio_data, sr = sf.read(input_path)
                        sf.write(temp_wav_path if denoise else wav_path, audio_data, sr)
                    elif input_format == "mp3":
                        output_target = temp_wav_path if denoise else wav_path
                        ffmpeg_cmd = [
                            "ffmpeg", "-y", "-i", input_path,
                            "-ac", "1", "-ar", "16000",  # mono, 16kHz for speech processing
                            output_target
                        ]
                        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    # elif input_format == "mp3":
                    #     audio = AudioSegment.from_mp3(input_path)
                    #     audio.export(temp_wav_path if denoise else wav_path, format="wav")
                    else:
                        print(f"❌ Unsupported format: {input_format}")
                        return

                    # Optional: Apply noise reduction
                    if denoise:
                        y, sr = librosa.load(temp_wav_path, sr=None)
                        noise_sample = y[:int(0.5 * sr)]  # 0.5 seconds for noise profile
                        reduced_y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
                        sf.write(wav_path, reduced_y, sr)
                        os.remove(temp_wav_path)  # Cleanup

                    print(f"✔ Converted: {input_path} → {wav_path}")
                    count += 1

                except Exception as e:
                    print(f"❌ Failed to convert {input_path}: {e}")

    print(f"\n✅ Done. {count} files converted.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio files to WAV under a base directory")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing folders with audio files")
    parser.add_argument("--input_format", type=str, choices=["flac", "mp3"], required=True,
                        help="Input audio format to convert (flac or mp3)")
    parser.add_argument("--denoise", action="store_true",
                        help="Apply noise reduction after conversion")
    args = parser.parse_args()

    convert_audio_to_wav(args.base_dir, args.input_format, args.denoise)

# Example Usage:
# python convert_audio_to_wav.py --base_dir path/to/Sachs --input_format mp3
# python convert_audio_to_wav.py --base_dir path/to/Sachs --input_format mp3 --denoise
