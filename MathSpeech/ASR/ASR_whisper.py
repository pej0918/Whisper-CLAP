import os
import torch
import pandas as pd
import whisper

# =========================
# Paths
# =========================
excel_path = "/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx"
audio_dir = "/data1/eunju/datasets/mathspeech/dataset"

save_path1 = "/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/whisper_base/result_ASR.csv"
save_path2 = "/home/pej0918/Projects/Audio_Text/MathSpeech/Ablation_Study/whisper_base/result_ASR.csv"

# =========================
# Load data
# =========================
df = pd.read_excel(excel_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# =========================
# Load Whisper models
# =========================
whisper_model_base = whisper.load_model("base").to(device)
whisper_model_medium = whisper.load_model("medium").to(device)

originalNT = df["transcription"]

base_result_list = []
medium_result_list = []

# =========================
# Run ASR
# =========================
for i in range(len(df)):
    try:
        audio_path = os.path.join(audio_dir, f"{i+1}.mp3")

        if not os.path.exists(audio_path):
            print(f"[missing] {audio_path}")
            base_result_list.append("")
            medium_result_list.append("")
            continue

        result_base = whisper_model_base.transcribe(
            audio_path,
            language="en",
            fp16=torch.cuda.is_available()
        )
        raw_base = result_base["text"].strip()
        print(f"whisper base result : {raw_base}")
        base_result_list.append(raw_base)

        result_medium = whisper_model_medium.transcribe(
            audio_path,
            language="en",
            fp16=torch.cuda.is_available()
        )
        raw_medium = result_medium["text"].strip()
        print(f"whisper medium result : {raw_medium}")
        medium_result_list.append(raw_medium)

        print(f"original NT text : {originalNT.iloc[i]}")
        print(f"================================= epoch {i+1} finish =================================")

    except Exception as e:
        print(f"[error] index {i+1}: {e}")
        base_result_list.append("")
        medium_result_list.append("")

# =========================
# Save results
# =========================
df["whisper_base_predSE"] = base_result_list
df["whisper_medium_predSE"] = medium_result_list

df.to_csv(save_path1, index=False)
df.to_csv(save_path2, index=False)

print(f"Saved to: {save_path1}")
print(f"Saved to: {save_path2}")