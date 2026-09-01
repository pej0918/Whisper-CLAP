import argparse
import hashlib
import json
import sys
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


def resolve_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError(
        "Could not find CLAP state_dict. Expected checkpoint['state_dict'], "
        "checkpoint['model_state_dict'], or a raw state_dict."
    )


def format_text(text, template):
    text = " ".join(str(text).strip().split())
    if template == "raw":
        return text
    if template == "lecturer_says":
        return f'The lecturer says: "{text}"'
    raise ValueError(template)


def sha1_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def task_paths(data_root, task):
    data_root = Path(data_root)
    if task == "sent":
        task_dir = data_root / "s2l_sent"
        prefix = "s2l_sent"
    elif task == "eq":
        task_dir = data_root / "s2l_eq"
        prefix = "s2l_eq"
    else:
        raise ValueError(task)

    manifest = task_dir / f"{prefix}_all.csv"
    output = task_dir / f"{prefix}_clap_text_emb.pt"
    metadata = task_dir / f"{prefix}_clap_text_emb_metadata.json"
    return task_dir, prefix, manifest, output, metadata


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Precompute Lecture-CLAP text embeddings for S2L spoken-math ASR. "
            "Embeddings are stored in global sample_id order so the existing "
            "Whisper-CLAP trainer can index them with sample_id - 1."
        )
    )
    ap.add_argument("--task", choices=["sent", "eq"], required=True)
    ap.add_argument(
        "--data_root",
        default="/data1/eunju/datasets/speech2latex_asr_seed42",
    )
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--clap_repo", default=None)
    ap.add_argument("--text_col", default="reference_text")
    ap.add_argument(
        "--text_template",
        choices=["raw", "lecturer_says"],
        default="raw",
        help="raw matches the MathSpeech CLAP text-precompute convention.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--amodel", default="HTSAT-base")
    ap.add_argument("--output", default=None)
    ap.add_argument("--metadata", default=None)
    args = ap.parse_args()

    task_dir, prefix, manifest, default_output, default_metadata = task_paths(
        args.data_root, args.task
    )
    output = Path(args.output) if args.output else default_output
    metadata_path = Path(args.metadata) if args.metadata else default_metadata

    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    if not Path(args.ckpt_path).exists():
        raise FileNotFoundError(f"Missing CLAP checkpoint: {args.ckpt_path}")

    if args.clap_repo:
        src = str((Path(args.clap_repo).resolve() / "src"))
        if src not in sys.path:
            sys.path.insert(0, src)

    import laion_clap

    df = pd.read_csv(manifest)
    required = {"sample_id", args.text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {manifest}: {sorted(missing)}")

    # The current Whisper-CLAP trainer assumes:
    #     embedding index = sample_id - 1
    # Therefore the all.csv manifest must contain contiguous global IDs 1..N.
    ids = df["sample_id"].astype(int)
    if ids.duplicated().any():
        dup = ids[ids.duplicated()].head().tolist()
        raise ValueError(f"Duplicate sample_id values: {dup}")

    df = df.sort_values("sample_id").reset_index(drop=True)
    ids = df["sample_id"].astype(int).tolist()
    expected_ids = list(range(1, len(df) + 1))
    if ids != expected_ids:
        raise ValueError(
            "sample_id must be contiguous 1..N in the task all.csv because the "
            "trainer indexes CLAP embeddings with sample_id - 1."
        )

    raw_texts = df[args.text_col].fillna("").astype(str).tolist()
    texts = [format_text(x, args.text_template) for x in raw_texts]
    empty = [i + 1 for i, text in enumerate(texts) if not text]
    if empty:
        raise ValueError(f"Found {len(empty)} empty texts; first sample_ids={empty[:10]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("S2L CLAP TEXT EMBEDDING PRECOMPUTE")
    print("=" * 80)
    print("task          :", args.task)
    print("manifest      :", manifest)
    print("samples       :", len(df))
    print("text column   :", args.text_col)
    print("text template :", args.text_template)
    print("CLAP ckpt     :", args.ckpt_path)
    print("amodel        :", args.amodel)
    print("device        :", device)
    print("batch size    :", args.batch_size)
    print("output        :", output)
    print("example text  :", texts[0])

    clap_model = laion_clap.CLAP_Module(
        enable_fusion=False,
        amodel=args.amodel,
        device=device,
    )

    ckpt = torch_load_compat(args.ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        print("ckpt name     :", ckpt.get("name", "<not stored>"))
        print("ckpt epoch    :", ckpt.get("epoch", "<not stored>"))

    state_dict = resolve_state_dict(ckpt)
    missing_keys, unexpected_keys = clap_model.model.load_state_dict(
        state_dict, strict=False
    )
    print("missing keys  :", len(missing_keys))
    print("unexpected    :", len(unexpected_keys))
    if missing_keys:
        print("first missing :", missing_keys[:10])
    if unexpected_keys:
        print("first unexpected:", unexpected_keys[:10])

    if len(missing_keys) > 50 or len(unexpected_keys) > 50:
        raise RuntimeError(
            "Too many CLAP checkpoint key mismatches. Verify that --ckpt_path "
            "is the same Lecture-CLAP checkpoint used for MathSpeech."
        )

    clap_model.eval()
    for p in clap_model.model.parameters():
        p.requires_grad = False

    chunks = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), args.batch_size),
            desc=f"S2L-{args.task} CLAP text embeddings",
        ):
            batch_texts = texts[start : start + args.batch_size]
            emb = clap_model.get_text_embedding(batch_texts, use_tensor=True)
            if not torch.is_tensor(emb):
                emb = torch.tensor(emb)
            emb = F.normalize(emb.float().to(device), dim=-1)
            chunks.append(emb.cpu())

    embeddings = torch.cat(chunks, dim=0)
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(df):
        raise RuntimeError(
            f"Embedding count mismatch: embeddings={embeddings.shape[0]}, rows={len(df)}"
        )
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("CLAP embeddings contain NaN/Inf values")

    # IMPORTANT: save the tensor itself, not a metadata dict.
    # train_mathspeech_projector_best_wer.py currently calls:
    #     torch_load_compat(...).float()
    # and expects shape [N, D].
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings.contiguous(), output)

    metadata = {
        "task": args.task,
        "manifest": str(manifest.resolve()),
        "manifest_sha1": sha1_file(manifest),
        "num_samples": int(len(df)),
        "sample_id_min": int(ids[0]),
        "sample_id_max": int(ids[-1]),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "text_col": args.text_col,
        "text_template": args.text_template,
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "amodel": args.amodel,
        "l2_normalized": True,
        "indexing": "embedding[sample_id - 1]",
        "output": str(output.resolve()),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("embedding shape:", tuple(embeddings.shape))
    print("embedding dtype:", embeddings.dtype)
    print("mean L2 norm   :", embeddings.norm(dim=-1).mean().item())
    print("saved tensor   :", output)
    print("metadata       :", metadata_path)


if __name__ == "__main__":
    main()
