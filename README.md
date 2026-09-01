# Whisper-CLAP

Clean MathSpeech and Speech2Latex ASR experiments with source/group-aware splits, Hugging Face Whisper evaluation, validation-WER checkpoint selection, parameter-efficient baselines, and corpus-level jiwer metrics.

## What changed

- MathSpeech split is now **source-aware** with `seed=42`, preventing the same source from appearing in both train and test.
- Speech2Latex support was added for **English human-only** S2L-sentences and S2L-equations subsets.
- Whisper-base baseline and all model variants use the same **Hugging Face Transformers** Whisper pipeline.
- Training scripts save the best checkpoint using **validation WER** rather than validation loss.
- WER/CER are computed with official `jiwer` **corpus-level** metrics instead of averaging per-sample WER/CER.
- LoRA-Whisper and Residual Adapter-Whisper baselines were added.
- Old OpenAI-Whisper baseline and duplicate v2 scripts were removed in favor of clearer canonical filenames.

## Canonical files

| File | Purpose |
| --- | --- |
| `mathspeech_utils.py` | Shared seed, ffmpeg audio loading, and MathSpeech source-aware split utilities |
| `asr_dataset_utils.py` | Shared MathSpeech/Speech2Latex dataset loader and group-aware split utilities |
| `train_whisper_ft.py` | MathSpeech Hugging Face Whisper fine-tuning baseline; selects `best.pt` by validation WER |
| `train_whisper_ft_generic.py` | Generic Hugging Face Whisper full fine-tuning for MathSpeech or S2L |
| `train_whisper_lora.py` | LoRA-Whisper baseline with PEFT; CE-only; validation-WER checkpoint selection |
| `train_whisper_residual_adapter.py` | Residual Adapter-Whisper baseline; CE-only; validation-WER checkpoint selection |
| `train_whisper_clap_adapter.py` | Whisper + CLAP-guided adapter training; selects `best.pt` by validation WER |
| `eval_hf_whisper.py` | MathSpeech Hugging Face Whisper-base or fine-tuned Whisper evaluation |
| `eval_hf_whisper_generic.py` | Generic HF Whisper evaluation for MathSpeech or S2L |
| `eval_whisper_lora.py` | Evaluation for LoRA-Whisper adapter checkpoints |
| `eval_whisper_residual_adapter.py` | Evaluation for residual-adapter checkpoints |
| `eval_whisper_clap_adapter.py` | Evaluation for the CLAP-guided adapter checkpoint |
| `compute_asr_metrics.py` | Corpus-level WER/CER and MathSpeech auxiliary metrics |

## Requirements

```bash
pip install pandas openpyxl jiwer transformers torch tqdm peft datasets
```

`ffmpeg` must be available on `PATH` because local audio is decoded through ffmpeg. For Speech2Latex audio resampling, install at least one of:

```bash
pip install torchaudio
# or
pip install librosa
```

## MathSpeech: 4-method CLAP setting

```bash
bash bash/run_mathspeech_4methods_epoch10.sh
```

This runs:

- Whisper-base
- Whisper full fine-tuning
- CLAP-guided full fine-tuning
- Ours: CLAP-guided adapter

Common setting:

- max epoch 10
- learning rate 1e-5
- Whisper-base backbone
- source-aware split, seed 42
- validation-WER best checkpoint
- jiwer corpus-level WER/CER

## MathSpeech: LoRA and Residual Adapter baselines

```bash
bash bash/run_mathspeech_lora_residual_epoch10.sh
```

This adds:

- LoRA-Whisper: `r=16`, `lora_alpha=32`, `target_modules=q_proj,v_proj`, CE-only
- Residual Adapter-Whisper: bottleneck 256, Whisper frozen, CE-only

## Speech2Latex English Human subsets

```bash
bash bash/run_s2l_english_human_epoch10.sh
```

This uses:

- `marsianin500/Speech2Latex`
- S2L-sentences, English human only: `language=eng`, `is_tts=0`
- S2L-equations, English human only: `language=eng`, `is_tts=0`
- target column: `pronunciation`
- group-aware split by `sentence_id` when available, with fallback to equation/formula/text identifiers

It runs, for each S2L subset:

- Whisper-base
- Whisper full fine-tuning
- LoRA-Whisper
- Residual Adapter-Whisper

## Train CLAP-guided adapter on MathSpeech

```bash
python train_whisper_clap_adapter.py \
  --excel_path /data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx \
  --audio_dir /data1/eunju/datasets/mathspeech/dataset \
  --clap_emb_path /data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt \
  --save_dir /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter \
  --whisper_name openai/whisper-base \
  --source_col Source \
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

Then rerun with, for example, `--source_col Source`.

## Evaluate Hugging Face Whisper-base on MathSpeech

```bash
python eval_hf_whisper.py \
  --excel_path /data1/eunju/datasets/mathspeech/dataset/MathSpeech.xlsx \
  --audio_dir /data1/eunju/datasets/mathspeech/dataset \
  --split_path /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter/split_indices.pt \
  --save_csv /home/pej0918/Projects/Audio_Text/MathSpeech/Experiments/source_aware_clap_adapter/hf_whisper_base_test.csv \
  --eval_split test \
  --model_name openai/whisper-base
```

## Evaluate CLAP-guided adapter on MathSpeech

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

The reported `WER` and `CER` are corpus-level jiwer metrics. Training-time `valid_wer` uses the same normalization before jiwer corpus-level WER and is used for model selection.
