import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_emb(exp_dir, split="test"):
    path = os.path.join(exp_dir, f"alignment_embeddings_{split}.pt")
    obj = torch.load(path, map_location="cpu")

    print("Loaded:", path)
    print("method:", obj.get("method", "unknown"))
    print("keys:", obj.keys())

    audio_emb = obj["audio_embs"].float()
    text_emb = obj["text_embs"].float()
    scores = obj["scores"].float()

    audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

    return audio_emb.numpy(), text_emb.numpy(), scores.numpy()


def plot_tsne(exp_dir, title, save_path, split="test", max_lines=40):
    audio_emb, text_emb, scores = load_emb(exp_dir, split)

    n = audio_emb.shape[0]
    all_emb = np.concatenate([audio_emb, text_emb], axis=0)

    perplexity = min(30, max(5, n // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )

    xy = tsne.fit_transform(all_emb)
    audio_xy = xy[:n]
    text_xy = xy[n:]

    pair_dist = np.linalg.norm(audio_xy - text_xy, axis=1).mean()
    align_mean = scores.mean()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        audio_xy[:, 0],
        audio_xy[:, 1],
        s=28,
        alpha=0.75,
        label="Audio projector embedding",
    )
    plt.scatter(
        text_xy[:, 0],
        text_xy[:, 1],
        s=28,
        alpha=0.75,
        label="CLAP text embedding",
    )

    # 같은 sample의 audio/text pair 연결선
    for i in range(min(n, max_lines)):
        plt.plot(
            [audio_xy[i, 0], text_xy[i, 0]],
            [audio_xy[i, 1], text_xy[i, 1]],
            linewidth=0.5,
            alpha=0.25,
        )

    plt.title(
        f"{title}\n"
        f"Align score: {align_mean:.4f}, t-SNE pair dist.: {pair_dist:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("saved:", save_path)
    print("align score:", align_mean)
    print("mean t-SNE pair distance:", pair_dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce_dir", type=str, required=True)
    parser.add_argument("--ours_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    plot_tsne(
        args.ce_dir,
        "CE-only baseline",
        os.path.join(args.save_dir, "tsne_ce_only.png"),
        split=args.split,
    )

    plot_tsne(
        args.ours_dir,
        "Ours: CLAP-guided alignment",
        os.path.join(args.save_dir, "tsne_ours_alignment.png"),
        split=args.split,
    )


if __name__ == "__main__":
    main()