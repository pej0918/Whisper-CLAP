import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def format_text(text: str, template: str) -> str:
    text = " ".join(str(text).strip().split())
    if template == "raw":
        return text
    if template == "lecturer_says":
        # Matches prepare_clap_dataset.py found in the team's ZIP.
        return f'The lecturer says: "{text}"'
    raise ValueError(f"Unknown text template: {template}")


def resolve_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Some checkpoints are themselves a state_dict.
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError(
        "Could not find CLAP state_dict. Expected checkpoint['state_dict'], "
        "checkpoint['model_state_dict'], or a raw state_dict."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text_col", default="text_spoken")
    ap.add_argument(
        "--text_template",
        choices=["raw", "lecturer_says"],
        default="raw",
        help=(
            "raw reproduces the team's MathSpeech precompute script; "
            "lecturer_says matches the LPM Stage-1 caption template."
        ),
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--amodel", default="HTSAT-base")
    ap.add_argument(
        "--clap_repo",
        default=None,
        help="Optional LAION-CLAP repo root; <repo>/src is added to sys.path.",
    )
    args = ap.parse_args()

    if args.clap_repo:
        src = str(Path(args.clap_repo).resolve() / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

    import laion_clap

    df = pd.read_csv(args.manifest)
    required = ["sample_id", args.text_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing manifest columns: {missing_cols}")

    sample_ids = df["sample_id"].astype(str).tolist()
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id is not unique in the manifest.")

    raw_texts = df[args.text_col].fillna("").astype(str).tolist()
    texts = [format_text(t, args.text_template) for t in raw_texts]
    if any(t == "" for t in texts):
        n_empty = sum(t == "" for t in texts)
        raise ValueError(f"Found {n_empty} empty transcripts after formatting.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("manifest:", args.manifest)
    print("num texts:", len(texts))
    print("text template:", args.text_template)
    print("example:", texts[0] if texts else "<empty>")

    clap_model = laion_clap.CLAP_Module(
        enable_fusion=False,
        amodel=args.amodel,
        device=device,
    )

    ckpt = torch_load_compat(args.ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        print("ckpt name:", ckpt.get("name", "<not stored>"))
        print("ckpt epoch:", ckpt.get("epoch", "<not stored>"))

    state_dict = resolve_state_dict(ckpt)
    missing, unexpected = clap_model.model.load_state_dict(state_dict, strict=False)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))
    if missing:
        print("first missing:", missing[:10])
    if unexpected:
        print("first unexpected:", unexpected[:10])

    # A very large number of missing/unexpected keys usually means the wrong checkpoint.
    if len(missing) > 50 or len(unexpected) > 50:
        raise RuntimeError(
            "Too many CLAP checkpoint key mismatches. Verify that --ckpt_path is "
            "the Lecture-CLAP checkpoint used for Stage 1."
        )

    clap_model.eval()
    for p in clap_model.model.parameters():
        p.requires_grad = False

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="CLAP text embeddings"):
            batch_texts = texts[i:i + args.batch_size]
            emb = clap_model.get_text_embedding(batch_texts, use_tensor=True)
            if not torch.is_tensor(emb):
                emb = torch.tensor(emb)
            emb = F.normalize(emb.float().to(device), dim=-1)
            all_embs.append(emb.cpu())

    embeddings = torch.cat(all_embs, dim=0)
    if embeddings.shape[0] != len(df):
        raise RuntimeError("Embedding count mismatch after precompute.")

    payload = {
        "embeddings": embeddings,
        "sample_ids": sample_ids,
        "manifest": str(Path(args.manifest).resolve()),
        "text_col": args.text_col,
        "text_template": args.text_template,
        "ckpt_path": args.ckpt_path,
        "amodel": args.amodel,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    print("final embedding shape:", tuple(embeddings.shape))
    print("saved to:", out)


if __name__ == "__main__":
    main()
