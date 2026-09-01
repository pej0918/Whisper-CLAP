import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Input: 직접 값 넣는 버전
# =========================
data = [
    {
        "Method": "Residual CE-only",
        "AlignScore": -0.0186,
    },
    {
        "Method": "Residual + Cosine",
        "AlignScore": 0.7704,
    },
    {
        "Method": "Residual + MSE",
        "AlignScore": 0.7574,
    },
    {
        "Method": "Residual + CLIP",
        "AlignScore": 0.3302,
    },
    {
        "Method": "Residual + Cosine+CLIP",
        "AlignScore": 0.7523,
    },
    {
        "Method": "Gated Mean + Cosine",
        "AlignScore": 0.7669,
    },
    {
        "Method": "Gated Attn + Cosine",
        "AlignScore": 0.8500,
    },
    {
        "Method": "Gated CLS + Cosine",
        "AlignScore": 0.7289,
    },
]

df = pd.DataFrame(data)

save_dir = "/home/pej0918/Projects/Audio_Text/MathSpeech/figures"
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(10, 4.5))
plt.bar(df["Method"], df["AlignScore"])
plt.ylabel("Alignment Score")
plt.xlabel("Method")
plt.xticks(rotation=35, ha="right")
plt.ylim(-0.1, 0.95)
plt.tight_layout()

save_path = os.path.join(save_dir, "alignment_score_bar.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("saved:", save_path)