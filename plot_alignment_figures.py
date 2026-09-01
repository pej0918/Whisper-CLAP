# plot_alignment_figures.py
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_alignment_pt(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    audio = obj["audio_embs"].float()
    text = obj["text_embs"].float()
    scores = obj["scores"].float()

    method = obj.get("method", os.path.basename(os.path.dirname(path)))
    texts = obj.get("texts", None)

    return {
        "method": method,
        "audio": audio,
        "text": text,
        "scores": scores,
        "texts": texts,
    }


def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def save_alignment_bar_and_box(data_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    labels = list(data_dict.keys())
    means = [data_dict[k]["scores"].mean().item() for k in labels]
    stds = [data_dict[k]["scores"].std().item() for k in labels]

    # Bar plot
    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean alignment score")
    plt.title("Alignment score comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alignment_barplot.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "alignment_barplot.pdf"))
    plt.close()

    # Boxplot
    plt.figure(figsize=(10, 5))
    score_lists = [data_dict[k]["scores"].numpy() for k in labels]
    plt.boxplot(score_lists, labels=labels, showmeans=True)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Per-sample cosine similarity")
    plt.title("Alignment score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alignment_boxplot.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "alignment_boxplot.pdf"))
    plt.close()


def save_alignment_histogram(data_dict, save_dir, selected=None):
    os.makedirs(save_dir, exist_ok=True)

    if selected is None:
        selected = list(data_dict.keys())

    plt.figure(figsize=(9, 5))

    for label in selected:
        scores = data_dict[label]["scores"].numpy()
        mean = scores.mean()
        plt.hist(
            scores,
            bins=25,
            alpha=0.55,
            label=f"{label}, mean={mean:.3f}",
        )

    plt.xlabel("Cosine similarity with CLAP text embedding")
    plt.ylabel("Number of samples")
    plt.title("Alignment score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alignment_histogram.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "alignment_histogram.pdf"))
    plt.close()


def save_similarity_heatmap(data, save_dir, name, num_samples=20):
    os.makedirs(save_dir, exist_ok=True)

    audio = l2_normalize(data["audio"])
    text = l2_normalize(data["text"])

    n = min(num_samples, audio.size(0))
    audio = audio[:n]
    text = text[:n]

    sim = audio @ text.T
    sim_np = sim.numpy()

    diag_mean = np.diag(sim_np).mean()
    off_diag = sim_np[~np.eye(n, dtype=bool)]
    off_diag_mean = off_diag.mean()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(sim_np, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("CLAP text embedding index")
    plt.ylabel("Audio projector embedding index")
    plt.title(
        f"{name}\n"
        f"diag={diag_mean:.3f}, off-diag={off_diag_mean:.3f}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"similarity_heatmap_{name}.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"similarity_heatmap_{name}.pdf"))
    plt.close()


def save_pca_or_umap(data_dict, save_dir, selected_pair, method="umap"):
    """
    selected_pair: tuple like ("CE-only", "Ours")
    method: "umap" or "pca"
    """
    os.makedirs(save_dir, exist_ok=True)

    label = selected_pair
    data = data_dict[label]

    audio = l2_normalize(data["audio"]).numpy()
    text = l2_normalize(data["text"]).numpy()

    X = np.concatenate([audio, text], axis=0)
    n = audio.shape[0]

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            Z = reducer.fit_transform(X)
            plot_name = "UMAP"
        except ImportError:
            print("[Warning] umap-learn not installed. Falling back to PCA.")
            from sklearn.decomposition import PCA
            Z = PCA(n_components=2, random_state=42).fit_transform(X)
            plot_name = "PCA"
    else:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=42).fit_transform(X)
        plot_name = "PCA"

    Za = Z[:n]
    Zt = Z[n:]

    pair_dist = np.linalg.norm(Za - Zt, axis=1).mean()
    align_score = data["scores"].mean().item()

    plt.figure(figsize=(7, 6))
    plt.scatter(Za[:, 0], Za[:, 1], marker="o", alpha=0.75, label="Audio projector embedding")
    plt.scatter(Zt[:, 0], Zt[:, 1], marker="x", alpha=0.75, label="CLAP text embedding")

    # 너무 복잡하지 않게 일부 pair만 연결
    for i in range(min(25, n)):
        plt.plot(
            [Za[i, 0], Zt[i, 0]],
            [Za[i, 1], Zt[i, 1]],
            alpha=0.25,
            linewidth=0.8,
        )

    plt.title(
        f"{label}: {plot_name} cross-modal alignment\n"
        f"Align score={align_score:.4f}, projected pair dist={pair_dist:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{plot_name.lower()}_{label}.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{plot_name.lower()}_{label}.pdf"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/figures/alignment_figures",
    )
    parser.add_argument("--num_heatmap_samples", type=int, default=20)
    parser.add_argument("--embed_file", type=str, default="alignment_embeddings_test.pt")
    parser.add_argument("--reduction", type=str, default="umap", choices=["umap", "pca"])

    args = parser.parse_args()

    # 네 실험 이름에 맞게 label 정리
    experiments = {
        "Residual CE-only": "v5_arch_residual_mlp_ce",
        "Gated CE-only": "v5_arch_gated_ce",
        "Residual cosine": "v5_loss_residual_cosine",
        "Residual MSE": "v5_loss_residual_mse",
        "Residual CLIP": "v5_loss_residual_clip",
        "Residual cosine+CLIP": "v5_loss_residual_cosine_clip",
        "Gated mean cosine": "v5_pool_gated_mean_cosine",
        "Gated attn cosine": "v5_pool_gated_attn_cosine",
        "Gated cls cosine": "v5_pool_gated_cls_cosine",
    }

    data_dict = {}

    for label, exp in experiments.items():
        path = os.path.join(args.base_dir, exp, args.embed_file)

        if not os.path.exists(path):
            print(f"[Skip] missing: {path}")
            continue

        data = load_alignment_pt(path)
        data_dict[label] = data

        print(
            f"{label:25s} | "
            f"mean={data['scores'].mean().item():.4f} | "
            f"std={data['scores'].std().item():.4f}"
        )

    if len(data_dict) == 0:
        raise RuntimeError("No alignment embedding files found.")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. barplot + boxplot
    save_alignment_bar_and_box(data_dict, args.save_dir)

    # 2. histogram: 핵심 비교만
    selected_hist = [
        "Gated CE-only",
        "Gated attn cosine",
        "Gated cls cosine",
    ]
    selected_hist = [x for x in selected_hist if x in data_dict]
    save_alignment_histogram(data_dict, args.save_dir, selected=selected_hist)

    # 3. heatmap: CE-only vs Ours 비교
    for label in ["Gated CE-only", "Gated attn cosine", "Gated cls cosine", "Residual CE-only", "Residual cosine"]:
        if label in data_dict:
            safe_name = label.replace(" ", "_").replace("+", "plus")
            save_similarity_heatmap(
                data_dict[label],
                args.save_dir,
                safe_name,
                num_samples=args.num_heatmap_samples,
            )

    # 4. UMAP/PCA: 핵심 모델 각각
    for label in ["Gated CE-only", "Gated attn cosine", "Gated cls cosine"]:
        if label in data_dict:
            safe_label = label.replace(" ", "_").replace("+", "plus")
            temp_dict = {safe_label: data_dict[label]}
            save_pca_or_umap(
                temp_dict,
                args.save_dir,
                selected_pair=safe_label,
                method=args.reduction,
            )

    print()
    print("Saved figures to:", args.save_dir)


if __name__ == "__main__":
    main()