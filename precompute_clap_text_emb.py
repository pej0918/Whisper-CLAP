import os
import torch
import pandas as pd
from tqdm import tqdm
import laion_clap

# =========================
# Paths
# =========================
excel_path = "/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx"
ckpt_path = "/data1/dohee/model_ckpt/clap_finetuning_best.pt"
save_path = "/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# =========================
# Load MathSpeech transcripts
# =========================
df = pd.read_excel(excel_path)
texts = df["transcription"].astype(str).tolist()

print("num texts:", len(texts))
print("example:", texts[0])

# =========================
# Load fine-tuned Lecture-CLAP
# =========================
clap_model = laion_clap.CLAP_Module(
    enable_fusion=False, amodel= 'HTSAT-base',
    device=device
)

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False,
)

print("ckpt name:", ckpt["name"])
print("ckpt epoch:", ckpt["epoch"])

state_dict = ckpt["state_dict"]

missing, unexpected = clap_model.model.load_state_dict(
    state_dict,
    strict=False,
)

print("missing:", len(missing))
print("unexpected:", len(unexpected))

if len(missing) > 0:
    print("first missing:", missing[:20])

if len(unexpected) > 0:
    print("first unexpected:", unexpected[:20])

clap_model.eval()

for p in clap_model.model.parameters():
    p.requires_grad = False

# =========================
# Precompute text embeddings
# =========================
all_embs = []
batch_size = 32

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        emb = clap_model.get_text_embedding(
            batch_texts,
            use_tensor=True,
        )

        if not torch.is_tensor(emb):
            emb = torch.tensor(emb)

        emb = emb.float().to(device)
        emb = torch.nn.functional.normalize(emb, dim=-1)

        all_embs.append(emb.cpu())

all_embs = torch.cat(all_embs, dim=0)

print("final embedding shape:", all_embs.shape)

os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(all_embs, save_path)

print("saved to:", save_path)