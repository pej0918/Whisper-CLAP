import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import whisper as openai_whisper

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


# =========================================================
# Dataset
# =========================================================
class MathSpeechDataset(Dataset):
    def __init__(
        self,
        df,
        indices,
        audio_dir,
        processor,
        clap_embs=None,
    ):
        self.df = df.reset_index(drop=True)
        self.indices = list(indices)
        self.audio_dir = audio_dir
        self.processor = processor
        self.clap_embs = clap_embs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        text = str(row["transcription"])
        audio_path = os.path.join(self.audio_dir, f"{real_idx + 1}.mp3")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        # OpenAI whisper loader: 16kHz mono float32 numpy
        audio = openai_whisper.load_audio(audio_path)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]

        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
        ).input_ids[0]

        item = {
            "input_features": input_features,
            "labels": labels,
            "text": text,
            "real_idx": real_idx,
        }

        if self.clap_embs is not None:
            item["clap_emb"] = self.clap_embs[real_idx]

        return item


class DataCollatorSpeechSeq2SeqWithClap:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_features = torch.stack(
            [b["input_features"] for b in batch],
            dim=0,
        )

        label_features = [{"input_ids": b["labels"]} for b in batch]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]

        # Padding token은 CE loss에서 무시
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Whisper training 관례: decoder_start_token_id가 붙어 있으면 제거
        decoder_start_token_id = self.processor.tokenizer.bos_token_id
        if labels.shape[1] > 0 and (labels[:, 0] == decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        output = {
            "input_features": input_features,
            "labels": labels,
            "texts": [b["text"] for b in batch],
            "real_indices": [b["real_idx"] for b in batch],
        }

        if "clap_emb" in batch[0]:
            output["clap_emb"] = torch.stack(
                [b["clap_emb"] for b in batch],
                dim=0,
            )

        return output


# =========================================================
# Model
# =========================================================
class ResidualSemanticAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=256, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

        # 처음에는 adapter 영향력을 작게 시작
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h):
        return h + self.scale * self.net(h)


class WhisperSemanticASR(nn.Module):
    def __init__(
        self,
        whisper_name="openai/whisper-base",
        clap_dim=512,
        adapter_bottleneck=256,
        freeze_whisper=True,
    ):
        super().__init__()

        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        self.whisper.config.use_cache = False

        hidden_dim = self.whisper.config.d_model

        self.adapter = ResidualSemanticAdapter(
            hidden_dim=hidden_dim,
            bottleneck_dim=adapter_bottleneck,
        )

        self.align_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, clap_dim),
        )

        if freeze_whisper:
            for p in self.whisper.parameters():
                p.requires_grad = False

        # adapter와 align_head는 항상 학습
        for p in self.adapter.parameters():
            p.requires_grad = True

        for p in self.align_head.parameters():
            p.requires_grad = True

    def encode_with_adapter(self, input_features):
        encoder_outputs = self.whisper.model.encoder(input_features)
        h = encoder_outputs.last_hidden_state

        h_adapted = self.adapter(h)
        return h_adapted

    def forward(
        self,
        input_features,
        labels,
        clap_emb=None,
        lambda_align=0.1,
    ):
        h_adapted = self.encode_with_adapter(input_features)

        adapted_encoder_outputs = BaseModelOutput(
            last_hidden_state=h_adapted,
        )

        outputs = self.whisper(
            encoder_outputs=adapted_encoder_outputs,
            labels=labels,
            return_dict=True,
        )

        ce_loss = outputs.loss
        total_loss = ce_loss

        align_loss = None

        if clap_emb is not None and lambda_align > 0:
            # MVP: encoder hidden state mean pooling
            pooled = h_adapted.mean(dim=1)
            z = self.align_head(pooled)

            z = F.normalize(z, dim=-1)
            clap_emb = F.normalize(clap_emb, dim=-1)

            align_loss = 1.0 - F.cosine_similarity(
                z,
                clap_emb,
                dim=-1,
            ).mean()

            total_loss = ce_loss + lambda_align * align_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "align_loss": align_loss.detach() if align_loss is not None else None,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(self, input_features, **kwargs):
        h_adapted = self.encode_with_adapter(input_features)

        adapted_encoder_outputs = BaseModelOutput(
            last_hidden_state=h_adapted,
        )

        pred_ids = self.whisper.generate(
            encoder_outputs=adapted_encoder_outputs,
            **kwargs,
        )

        return pred_ids


# =========================================================
# Utils
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_split(n, seed=42, train_ratio=0.8, valid_ratio=0.1):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    return train_idx, valid_idx, test_idx


@torch.no_grad()
def validate(model, loader, device, lambda_align):
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_align = 0.0
    align_count = 0

    for batch in tqdm(loader, desc="valid"):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        clap_emb = batch.get("clap_emb", None)
        if clap_emb is not None:
            clap_emb = clap_emb.to(device)

        out = model(
            input_features=input_features,
            labels=labels,
            clap_emb=clap_emb,
            lambda_align=lambda_align,
        )

        total_loss += out["loss"].item()
        total_ce += out["ce_loss"].item()

        if out["align_loss"] is not None:
            total_align += out["align_loss"].item()
            align_count += 1

    avg_loss = total_loss / max(len(loader), 1)
    avg_ce = total_ce / max(len(loader), 1)
    avg_align = total_align / max(align_count, 1)

    return avg_loss, avg_ce, avg_align


# =========================================================
# Train
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--excel_path",
        type=str,
        default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/data1/eunju/datasets/mathspeech/dataset",
    )
    parser.add_argument(
        "--clap_emb_path",
        type=str,
        default="/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/whisper_base_projector_align",
    )

    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--adapter_bottleneck", type=int, default=256)
    parser.add_argument("--freeze_whisper", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_excel(args.excel_path)
    n = len(df)
    print("num samples:", n)

    train_idx, valid_idx, test_idx = make_split(n, seed=args.seed)

    split_path = os.path.join(args.save_dir, "split_indices.pt")
    torch.save(
        {
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "seed": args.seed,
        },
        split_path,
    )
    print("saved split to:", split_path)
    print(f"split sizes: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")

    processor = WhisperProcessor.from_pretrained(
        args.whisper_name,
        language="en",
        task="transcribe",
    )

    # CLAP embedding load
    clap_embs = None
    clap_dim = 512

    if args.lambda_align > 0:
        if args.clap_emb_path is None or not os.path.exists(args.clap_emb_path):
            raise FileNotFoundError(
                f"lambda_align > 0 but clap_emb_path not found: {args.clap_emb_path}"
            )

        clap_embs = torch.load(args.clap_emb_path, map_location="cpu").float()
        print("loaded clap embs:", clap_embs.shape)

        if clap_embs.shape[0] != n:
            raise ValueError(
                f"CLAP embedding count mismatch: clap_embs={clap_embs.shape[0]}, dataset={n}"
            )

        clap_dim = clap_embs.shape[-1]
    else:
        print("lambda_align=0.0, CE-only training")

    train_dataset = MathSpeechDataset(
        df=df,
        indices=train_idx,
        audio_dir=args.audio_dir,
        processor=processor,
        clap_embs=clap_embs,
    )

    valid_dataset = MathSpeechDataset(
        df=df,
        indices=valid_idx,
        audio_dir=args.audio_dir,
        processor=processor,
        clap_embs=clap_embs,
    )

    collator = DataCollatorSpeechSeq2SeqWithClap(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = WhisperSemanticASR(
        whisper_name=args.whisper_name,
        clap_dim=clap_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        freeze_whisper=args.freeze_whisper,
    ).to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",
        task="transcribe",
    )
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())

    print("total params:", num_total)
    print("trainable params:", num_trainable)
    print("freeze_whisper:", args.freeze_whisper)
    print("lambda_align:", args.lambda_align)
    print("clap_dim:", clap_dim)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_valid_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_align = 0.0
        align_count = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")

        for batch in pbar:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            clap_emb = batch.get("clap_emb", None)
            if clap_emb is not None:
                clap_emb = clap_emb.to(device)

            out = model(
                input_features=input_features,
                labels=labels,
                clap_emb=clap_emb,
                lambda_align=args.lambda_align,
            )

            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += out["loss"].item()
            total_ce += out["ce_loss"].item()

            if out["align_loss"] is not None:
                total_align += out["align_loss"].item()
                align_count += 1

            pbar.set_postfix(
                loss=f"{out['loss'].item():.4f}",
                ce=f"{out['ce_loss'].item():.4f}",
                align=f"{out['align_loss'].item():.4f}" if out["align_loss"] is not None else "0.0000",
            )

        avg_train_loss = total_loss / max(len(train_loader), 1)
        avg_train_ce = total_ce / max(len(train_loader), 1)
        avg_train_align = total_align / max(align_count, 1)

        valid_loss, valid_ce, valid_align = validate(
            model=model,
            loader=valid_loader,
            device=device,
            lambda_align=args.lambda_align,
        )

        print(
            f"[epoch {epoch}] "
            f"train_loss={avg_train_loss:.4f}, "
            f"train_ce={avg_train_ce:.4f}, "
            f"train_align={avg_train_align:.4f} | "
            f"valid_loss={valid_loss:.4f}, "
            f"valid_ce={valid_ce:.4f}, "
            f"valid_align={valid_align:.4f}"
        )

        last_path = os.path.join(args.save_dir, "last.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "clap_dim": clap_dim,
                "epoch": epoch,
                "valid_loss": valid_loss,
                "valid_ce": valid_ce,
                "valid_align": valid_align,
            },
            last_path,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "clap_dim": clap_dim,
                    "epoch": epoch,
                    "best_valid_loss": best_valid_loss,
                    "valid_ce": valid_ce,
                    "valid_align": valid_align,
                },
                best_path,
            )

            print("saved best:", best_path)


if __name__ == "__main__":
    main()
