# Whisper-CLAP

Research code for CLAP-guided parameter-efficient adaptation of Whisper for specialized mathematical ASR.

> **Active paper branch:** `common_clap_whisper`

## 1. Final common protocol

The final paper protocol is shared by MathSpeech, S2L-Sentences, and S2L-Equations.

```text
Backbone              : openai/whisper-base
Max epochs            : 10
LR search             : {1e-5, 1e-4, 3e-4}
Validation decoding   : beam=5
Checkpoint selection  : lowest validation WER within each LR
LR selection          : lowest validation WER across the 3 LR runs
Test decoding         : beam=5
Final report          : Test WER / CER of the validation-selected run
lambda_hidden         : 0.1 (CLAP-guided methods)
lambda_align          : 0.05 (CLAP-guided methods)
Seed                   : 42
```

The test set must not be used to choose the LR. The intended flow is:

```text
for each trainable method
    -> train LR={1e-5,1e-4,3e-4}, up to 10 epochs each
    -> within each LR, retain the checkpoint with best Valid WER @ beam=5
    -> compare the three best Valid WER values
    -> select the LR with the lowest Valid WER
    -> evaluate only that selected run on Test @ beam=5
    -> report Test WER / CER
```

`scripts/select_best_lr.py` implements the LR-selection step.

## 2. Methods

| Method | Whisper backbone | Trainable module | CLAP | Objective |
|---|---|---|:---:|---|
| Whisper-base (Zero-Shot) | Frozen | None | No | — |
| Full Fine-tuning | Trainable | Full Whisper | No | CE |
| LoRA-Whisper (+out_proj) | Frozen | LoRA | No | CE |
| LoRA-Whisper (-out_proj) | Frozen | LoRA | No | CE |
| Whisper + Residual Adapter | Frozen | Residual Adapter | No | CE |
| CLAP-guided Full Fine-tuning | Trainable | Whisper + Gated Adapter + Align Head | Yes | CE + 0.1 Hidden + 0.05 Align |
| Ours (CLAP-guided Adapter) | Frozen | Gated Adapter + Align Head | Yes | CE + 0.1 Hidden + 0.05 Align |

Ours uses a gated residual projector with bottleneck 256, `pool_type=cls` (the first Whisper encoder time step), dropout 0.1, adapter-scale initialization 0.01, cosine CLAP alignment, and a frozen Whisper backbone.

## 3. MathSpeech

Source-disjoint manifests are under:

```text
mathspeech/splits_source_balanced_seed42/
```

The low-level single-run launcher is:

```text
command/run_mathspeech_reported_configs.sh
```

It supports `STAGE=train`, `STAGE=test`, and `STAGE=full`. The finalized 3-LR search should normally use the high-level runner:

```bash
METHOD=ours GPUS="1 2 3" \
bash command/run_mathspeech_final_protocol.sh
```

Replace `ours` with:

```text
fullft
clap_fullft
lora_out
lora_noout
residual_b256
ours
rq5b_lora_r7
rq5b_residual_b128
```

The high-level runner trains the three LR runs in parallel, selects the LR strictly by validation WER, and evaluates only the selected run on test.

Current validation-selected MathSpeech LRs from the finalized search are:

| Method | Selected LR |
|---|---:|
| Full FT | `1e-5` |
| CLAP-guided Full FT | `1e-5` |
| LoRA +out_proj, r=32 | `1e-4` |
| LoRA -out_proj, r=32 | `3e-4` |
| Residual Adapter, b=256 | `1e-4` |
| Ours | `3e-4` |
| Parameter-matched LoRA, r=7 | `3e-4` |
| Parameter-matched Residual, b=128 | `3e-4` |

## 4. MathSpeech loss ablation (RQ4)

The final Ours LR is `3e-4`. Loss ablations therefore fix LR=`3e-4` and change only the objective.

```bash
METHOD=rq4_ce_only   LR=3e-4 GPU=1 STAGE=full bash command/run_mathspeech_reported_configs.sh
METHOD=rq4_ce_hidden LR=3e-4 GPU=2 STAGE=full bash command/run_mathspeech_reported_configs.sh
METHOD=rq4_ce_align  LR=3e-4 GPU=3 STAGE=full bash command/run_mathspeech_reported_configs.sh
METHOD=ours          LR=3e-4 GPU=4 STAGE=full bash command/run_mathspeech_reported_configs.sh
```

```text
CE only      : L = L_CE
CE + Hidden  : L = L_CE + 0.1 L_hidden
CE + Align   : L = L_CE + 0.05 L_align
Full (Ours)  : L = L_CE + 0.1 L_hidden + 0.05 L_align
```

## 5. Parameter-matched PEFT (RQ5-B)

Parameter-matched configurations are separate models and therefore receive their own LR search.

```text
LoRA       : r=7, alpha=7, -out_proj       ~0.817M trainable
Residual   : bottleneck=128                ~0.790M trainable
Ours       : gated + CLAP                  ~0.790M trainable
```

On MathSpeech all three are validation-selected at `3e-4`.

## 6. S2L final main data: Human-only

Legacy Mix/H/A preprocessing and launchers are retained for exploratory or augmentation experiments. The final main S2L setting uses only real human English speech:

```text
language == eng
is_tts == 0
duration <= 30 sec
```

Prepare new Human-only manifests in a separate root, leaving legacy Mix data untouched:

```bash
python scripts/prepare_s2l_human_only_main.py --task all
```

Default output:

```text
/data1/eunju/datasets/speech2latex_asr_human_seed42/
  s2l_sent/
    s2l_sent_all.csv
    s2l_sent_train.csv
    s2l_sent_valid.csv
    s2l_sent_test.csv
  s2l_eq/
    s2l_eq_all.csv
    s2l_eq_train.csv
    s2l_eq_valid.csv
    s2l_eq_test.csv
```

S2L-Sentences keeps the released official train/test boundary, then creates a sentence-group-disjoint validation split from official train. S2L-Equations uses a deterministic formula-disjoint reconstructed split because the released HF configuration does not expose the paper's exact sample indices.

## 7. S2L Lecture-CLAP embeddings

After Human-only preprocessing, precompute task-level Lecture-CLAP text embeddings. The existing script is compatible because the new Human-only `*_all.csv` files use contiguous global sample IDs.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_s2l_clap_text_emb.py \
  --task sent \
  --data_root /data1/eunju/datasets/speech2latex_asr_human_seed42 \
  --ckpt_path /data1/dohee/model_ckpt/clap_finetuning_best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/precompute_s2l_clap_text_emb.py \
  --task eq \
  --data_root /data1/eunju/datasets/speech2latex_asr_human_seed42 \
  --ckpt_path /data1/dohee/model_ckpt/clap_finetuning_best.pt
```

## 8. S2L final experiments

The low-level Human-only launcher is:

```text
command/run_s2l_human_config.sh
```

The canonical final runner performs the full validation-selected 3-LR protocol:

```bash
TASK=sent METHOD=ours GPUS="1 2 3" \
bash command/run_s2l_human_final_protocol.sh

TASK=eq METHOD=ours GPUS="1 2 3" \
bash command/run_s2l_human_final_protocol.sh
```

Use the same method names as MathSpeech. RQ4 ablations are fixed-LR runs and should use `run_s2l_human_config.sh` after the final Ours LR for that S2L task has been selected. Parameter-matched RQ5-B configurations (`rq5b_lora_r7`, `rq5b_residual_b128`) should each receive the same three-LR validation search.

## 9. Representation analysis

MathSpeech mechanism analysis is implemented by:

```text
scripts/analyze_mathspeech_representation.py
scripts/plot_mathspeech_representation_mechanism.py
```

Semantic-alignment shift is interpreted only for variants whose AlignHead was trained with the CLAP alignment objective. CE-only and CE+Hidden should report semantic alignment as `—`; hidden preservation and representation drift remain valid.

## 10. Repository organization

```text
command/
  run_mathspeech_reported_configs.sh    low-level MathSpeech run
  run_mathspeech_final_protocol.sh      canonical MathSpeech 3-LR protocol
  run_s2l_human_config.sh               low-level Human-only S2L run
  run_s2l_human_final_protocol.sh       canonical Human-only S2L 3-LR protocol

scripts/
  select_best_lr.py                     validation-only LR selection
  prepare_s2l_human_only_main.py        final Human-only S2L manifests
  precompute_s2l_clap_text_emb.py       Lecture-CLAP text embeddings
  ...                                   training/evaluation/analysis scripts
```

Legacy S2L Mix/H/A code remains available for augmentation/source-ablation studies and is intentionally not overwritten by the final Human-only pipeline.
