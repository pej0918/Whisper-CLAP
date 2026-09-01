# Whisper-CLAP

Clean MathSpeech ASR experiments with a source-aware split, Hugging Face Whisper evaluation, and corpus-level jiwer metrics.

## What changed

- MathSpeech split is now **source-aware** with `seed=42`, preventing the same source from appearing in both train and test.
- Whisper-base baseline and all model variants use the same **Hugging Face Transformers** Whisper pipeline.
- WER/CER are computed with official `jiwer` **corpus-level** metrics instead of averaging per-sample WER/CER.
- Old OpenAI-Whisper baseline and duplicate v2 scripts were removed in favor of clearer canonical filenames.

## Canonical files

| File | Purpose |
| --- | --- |
| `mathspeech_utils.py` | Shared seed, ffmpeg audio loading, and source-aware split utilities |
| `train_whisper_ft.py` | Hugging Face Whisper fine-tuning baseline |
| `train_whisper_clap_adapter.py` | Whisper + CLAP-guided adapter training |
| `eval_hf_whisper.py` | Hugging Face Whisper-base or fine-tuned Whisper evaluation |
| `eval_whisper_clap_adapter.py` | Evaluation for the CLAP-guided adapter checkpoint |
| `compute_asr_metrics.py` | Corpus-level WER/CER and MathSpeech auxiliary metrics |

## Requirements

```bash
pip install pandas openpyxl jiwer transformers torch tqdm
```

`ffmpeg` must be available on `PATH` because audio is decoded through ffmpeg.

## Train CLAP-guided adapter

```bash
python train_whisper_clap_adapter.py \
  --excel_path /data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx \
  --audio_dir /data1/eunju/datasets/mathspeech/dataset \
  --clap_emb_path /data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter \
  --whisper_name openai/whisper-base \
  --seed 42 \
  --force_new_split
```

If the source column cannot be inferred automatically, inspect the Excel columns and pass it explicitly:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_excel('/data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx')
print(df.columns.tolist())
PY
```

Then rerun with, for example, `--source_col source`.

## Evaluate Hugging Face Whisper-base

```bash
python eval_hf_whisper.py \
  --excel_path /data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx \
  --audio_dir /data1/eunju/datasets/mathspeech/dataset \
  --split_path /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter/split_indices.pt \
  --save_csv /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter/hf_whisper_base_test.csv \
  --eval_split test \
  --model_name openai/whisper-base
```

## Evaluate CLAP-guided adapter

```bash
python eval_whisper_clap_adapter.py \
  --excel_path /data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx \
  --audio_dir /data1/eunju/datasets/mathspeech/dataset \
  --ckpt_path /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter/best.pt \
  --save_csv /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter/clap_adapter_test.csv \
  --eval_split test
```

## Compute corpus-level WER/CER

```bash
python compute_asr_metrics.py \
  --csv /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter/hf_whisper_base_test.csv \
  --pred_col pred_hf_whisper_base_test \
  --ref_col transcription \
  --out_csv /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter/hf_whisper_base_test_metrics.csv
```

The reported `WER` and `CER` are corpus-level jiwer metrics.
