import os
import argparse
import torch
import matplotlib.pyplot as plt


def load_scores(exp_dir, split="test"):
    path = os.path.join(exp_dir, f"alignment_embeddings_{split}.pt")
    obj = torch.load(path, map_location="cpu")
    scores = obj["scores"].float().numpy()
    method = obj.get("method", os.path.basename(exp_dir))
    return scores, method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce_dir", type=str, required=True)
    parser.add_argument("--ours_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ce_scores, ce_method = load_scores(args.ce_dir, args.split)
    ours_scores, ours_method = load_scores(args.ours_dir, args.split)

    print("CE-only:", ce_method, ce_scores.mean(), ce_scores.std())
    print("Ours:", ours_method, ours_scores.mean(), ours_scores.std())

    plt.figure(figsize=(6, 4))
    plt.hist(ce_scores, bins=20, alpha=0.6, label=f"CE-only, mean={ce_scores.mean():.3f}")
    plt.hist(ours_scores, bins=20, alpha=0.6, label=f"Ours, mean={ours_scores.mean():.3f}")

    plt.xlabel("Cosine similarity with CLAP text embedding")
    plt.ylabel("Number of samples")
    plt.title("Alignment score distribution")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(args.save_dir, "alignment_score_hist.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("saved:", save_path)


if __name__ == "__main__":
    main()