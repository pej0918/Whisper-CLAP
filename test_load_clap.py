import torch
import laion_clap

ckpt_path = "/data1/dohee/model_ckpt/clap_finetuning_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False,
)

state_dict = ckpt["state_dict"]

print("ckpt name:", ckpt["name"])
print("ckpt epoch:", ckpt["epoch"])
print("num params:", len(state_dict))

clap_model = laion_clap.CLAP_Module(
    enable_fusion=False, amodel= 'HTSAT-base',
    device=device
)

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

texts = [
    "the eigenvalue of a matrix",
    "the derivative of sine x is cosine x",
]

with torch.no_grad():
    emb = clap_model.get_text_embedding(
        texts,
        use_tensor=True,
    )

print("embedding type:", type(emb))
print("embedding shape:", emb.shape)
print("embedding dtype:", emb.dtype)
print("embedding device:", emb.device)