import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm

import whisper as openai_whisper
from transformers import WhisperProcessor

from train_whisper_projector_v2 import WhisperSemanticASR

try:
    from train_whisper_projector_position_v2 import WhisperSemanticASRPosition
    POSITION_IMPORT_ERROR = None
except Exception as e:
    WhisperSemanticASRPosition = None
    POSITION_IMPORT_ERROR = e


def clean_asr_tail(text):
    text = str(text).strip()
    text = text.replace(" ,", ",").replace(" .", ".")
    text = " ".join(text.split())

    words = text.split()
    if len(words) <= 4:
        return text.strip(" ,.").strip()

    # 1) non-English / broken unicode tail 제거
    for i, w in enumerate(words):
        if i >= 3 and any(ord(ch) > 127 for ch in w):
            return " ".join(words[:i]).strip(" ,.").strip()

    # 2) nd / ndi / ndx / ndy 류 tail 제거
    cut_prefixes = ["nd", "ndi", "ndx", "ndy", "ndt", "ndp"]
    for i, w in enumerate(words):
        ww = w.lower().strip(".,!?;:")
        if i >= 3 and any(ww.startswith(p) for p in cut_prefixes):
            return " ".join(words[:i]).strip(" ,.").strip()

    # 3) 자주 붙는 hallucination phrase 제거
    tail_phrases = [
        " more than",
        " more ",
        " of r",
        " of l",
        " of y",
        " of pi",
        " going to be",
        " it is",
        " it's",
        " the same as",
        " right here",
        " okay",
    ]

    lowered = " " + text.lower()
    best_cut = None

    for phrase in tail_phrases:
        pos = lowered.find(phrase)
        if pos != -1:
            prefix = text[:max(pos - 1, 0)].strip()
            if len(prefix.split()) >= 3:
                if best_cut is None or pos < best_cut:
                    best_cut = pos

    if best_cut is not None:
        return text[:best_cut - 1].strip(" ,.").strip()

    # 4) phrase 반복 제거
    words = text.split()
    norm_words = [w.lower().strip(".,!?;:") for w in words]

    for n in [5, 4, 3, 2, 1]:
        if len(words) < n * 3:
            continue

        for i in range(len(words) - 3 * n + 1):
            p1 = norm_words[i:i + n]
            p2 = norm_words[i + n:i + 2 * n]
            p3 = norm_words[i + 2 * n:i + 3 * n]

            if p1 == p2 == p3:
                return " ".join(words[:i + n]).strip(" ,.").strip()

    return text.strip(" ,.").strip()


def build_model_from_ckpt(ckpt, args):
    ckpt_args = ckpt.get("args", {})
    clap_dim = ckpt.get("clap_dim", 512)

    whisper_name = ckpt_args.get("whisper_name", args.whisper_name)
    adapter_type = ckpt_args.get("adapter_type", "residual_mlp")
    pool_type = ckpt_args.get("pool_type", "mean")
    adapter_bottleneck = ckpt_args.get("adapter_bottleneck", 256)
    dropout = ckpt_args.get("dropout", 0.1)
    adapter_scale_init = ckpt_args.get("adapter_scale_init", 0.01)

    adapter_position = ckpt_args.get("adapter_position", None)

    if adapter_position is None:
        print("[model type] standard post-encoder model")

        model = WhisperSemanticASR(
            whisper_name=whisper_name,
            clap_dim=clap_dim,
            adapter_type=adapter_type,
            pool_type=pool_type,
            adapter_bottleneck=adapter_bottleneck,
            dropout=dropout,
            adapter_scale_init=adapter_scale_init,
            freeze_whisper=True,
        )

    else:
        print("[model type] position-ablation model")
        print("adapter_position:", adapter_position)
        print("encoder_layer:", ckpt_args.get("encoder_layer", 6))

        if WhisperSemanticASRPosition is None:
            raise ImportError(
                "This checkpoint uses adapter_position, but "
                "train_whisper_projector_position_v2.py could not be imported.\n"
                f"Original import error: {POSITION_IMPORT_ERROR}"
            )

        model = WhisperSemanticASRPosition(
            whisper_name=whisper_name,
            clap_dim=clap_dim,
            adapter_type=adapter_type,
            pool_type=pool_type,
            adapter_position=adapter_position,
            encoder_layer=ckpt_args.get("encoder_layer", 6),
            adapter_bottleneck=adapter_bottleneck,
            dropout=dropout,
            adapter_scale_init=adapter_scale_init,
            freeze_whisper=True,
        )

    return model, whisper_name, ckpt_args


def make_pred_col(ckpt_args, eval_split):
    adapter_type = ckpt_args.get("adapter_type", "adapter")
    loss_type = ckpt_args.get("align_loss_type", "loss")
    pool_type = ckpt_args.get("pool_type", "pool")

    adapter_position = ckpt_args.get("adapter_position", None)

    if adapter_position is None:
        return f"pred_{adapter_type}_{pool_type}_{loss_type}_{eval_split}"

    encoder_layer = ckpt_args.get("encoder_layer", "none")
    return f"pred_{adapter_type}_{pool_type}_{loss_type}_{adapter_position}_L{encoder_layer}_{eval_split}"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel_path", type=str, default="/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx")
    parser.add_argument("--audio_dir", type=str, default="/data1/eunju/datasets/mathspeech/dataset")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-base")

    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--eval_split", type=str, default="test", choices=["all", "train", "valid", "test"])
    parser.add_argument("--pred_col", type=str, default=None)

    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    parser.add_argument("--no_clean", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_excel(args.excel_path)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    print("ckpt:", args.ckpt_path)
    print("ckpt epoch:", ckpt.get("epoch", "unknown"))

    model, whisper_name, ckpt_args = build_model_from_ckpt(ckpt, args)

    processor = WhisperProcessor.from_pretrained(
        whisper_name,
        language="en",
        task="transcribe",
    )

    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )

    print("missing keys:", len(missing))
    if len(missing) > 0:
        print("first missing:", missing[:5])

    print("unexpected keys:", len(unexpected))
    if len(unexpected) > 0:
        print("first unexpected:", unexpected[:5])

    model = model.to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",
        task="transcribe",
    )
    model.whisper.config.forced_decoder_ids = forced_decoder_ids
    model.whisper.generation_config.forced_decoder_ids = forced_decoder_ids

    if args.eval_split == "all":
        eval_indices = list(range(len(df)))
    else:
        split_path = os.path.join(os.path.dirname(args.ckpt_path), "split_indices.pt")
        split = torch.load(split_path, map_location="cpu", weights_only=False)
        eval_indices = split[f"{args.eval_split}_idx"]

    print("eval_split:", args.eval_split)
    print("num eval samples:", len(eval_indices))

    if args.pred_col is None:
        args.pred_col = make_pred_col(ckpt_args, args.eval_split)

    print("prediction column:", args.pred_col)
    print("clean prediction:", not args.no_clean)

    pred_dict = {}

    for real_idx in tqdm(eval_indices):
        audio_path = os.path.join(args.audio_dir, f"{real_idx + 1}.mp3")

        if not os.path.exists(audio_path):
            print("[missing]", audio_path)
            pred_dict[real_idx] = ""
            continue

        try:
            audio = openai_whisper.load_audio(audio_path)

            input_features = processor.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device)

            gen_kwargs = {
                "input_features": input_features,
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
                "do_sample": False,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "pad_token_id": processor.tokenizer.pad_token_id,
            }

            if args.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = args.repetition_penalty

            if args.no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

            pred_ids = model.generate(**gen_kwargs)

            pred = processor.batch_decode(
                pred_ids,
                skip_special_tokens=True,
            )[0].strip()

            raw_pred = pred

            if not args.no_clean:
                pred = clean_asr_tail(pred)

            pred_dict[real_idx] = pred

            print()
            print(f"[{real_idx + 1}]")
            print("GT       :", df["transcription"].iloc[real_idx])
            print("RAW PRED :", raw_pred)
            print("PRED     :", pred)

        except Exception as e:
            print(f"[error] index {real_idx + 1}: {e}")
            pred_dict[real_idx] = ""

    df[args.pred_col] = ""

    for idx, pred in pred_dict.items():
        df.loc[idx, args.pred_col] = pred

    save_dir = os.path.dirname(args.save_csv)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    df.to_csv(args.save_csv, index=False)

    print("saved to:", args.save_csv)
    print("prediction column:", args.pred_col)


if __name__ == "__main__":
    main()