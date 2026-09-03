# Whisper-CLAP

Research code for CLAP-guided parameter-efficient adaptation of Whisper for specialized ASR.

> **Active paper branch:** `common_clap_whisper`
>
> The canonical MathSpeech entry point for the currently reported configurations is
> `command/run_mathspeech_reported_configs.sh`.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate whisper_domain
```

or install from:

```bash
pip install -r requirements.txt
```

The current code uses PyTorch 2.1.2 + CUDA 12.1, Transformers 4.32.1, PEFT 0.5.0,
and Matplotlib for representation-analysis figures.

## 2. MathSpeech setup

The source-disjoint MathSpeech splits used by the current experiments are under:

```text
mathspeech/splits_source_balanced_seed42/
```

Projector-based methods expect precomputed Lecture-CLAP text embeddings. The default
launcher path is:

```text
/data1/eunju/datasets/mathspeech/dataset/mathspeech_clap_text_emb.pt
```

Override it when needed:

```bash
CLAP_EMB=/path/to/mathspeech_clap_text_emb.pt \
METHOD=ours GPU=1 \
bash command/run_mathspeech_reported_configs.sh
```

## 3. Canonical MathSpeech runs

Every trainable method is trained for 10 epochs. Within a learning-rate run, the
checkpoint with the lowest validation WER using beam 5 is selected and then evaluated
on the test split. `TEST_BEAM=5` is the default and can be overridden.

The current reported learning-rate configurations are:

| Method | Default LR | Trainable setup |
|---|---:|---|
| Whisper-base | — | frozen pretrained baseline |
| Full FT | `1e-5` | full Whisper fine-tuning |
| CLAP-guided Full FT | `1e-5` | Whisper + gated projector + semantic/hidden losses |
| LoRA -out_proj | `5e-4` | rank 32, q/k/v/fc1/fc2 |
| LoRA +out_proj | `5e-4` | rank 32, q/k/v/out/fc1/fc2 |
| Residual Adapter | `5e-4` | bottleneck 256 |
| Ours | `3e-4` | gated residual projector, frozen Whisper |

Run one configuration on any available GPU:

```bash
METHOD=whisper_base GPU=1 bash command/run_mathspeech_reported_configs.sh
METHOD=fullft GPU=1 bash command/run_mathspeech_reported_configs.sh
METHOD=clap_fullft GPU=1 bash command/run_mathspeech_reported_configs.sh
METHOD=lora_noout GPU=1 bash command/run_mathspeech_reported_configs.sh
METHOD=lora_out GPU=1 bash command/run_mathspeech_reported_configs.sh
METHOD=residual_b256 GPU=1 bash command/run_mathspeech_reported_configs.sh
METHOD=ours GPU=1 bash command/run_mathspeech_reported_configs.sh
```

The method-specific default LR can be overridden explicitly, e.g.:

```bash
METHOD=ours LR=5e-4 GPU=1 bash command/run_mathspeech_reported_configs.sh
```

## 4. Loss ablation (RQ4)

The current loss ablation uses the same gated architecture and a fixed LR of `3e-4`.

```bash
METHOD=rq4_ce_only   GPU=1 bash command/run_mathspeech_reported_configs.sh
METHOD=rq4_ce_hidden GPU=2 bash command/run_mathspeech_reported_configs.sh
METHOD=rq4_ce_align  GPU=3 bash command/run_mathspeech_reported_configs.sh
METHOD=ours          GPU=4 bash command/run_mathspeech_reported_configs.sh
```

Loss definitions:

```text
CE only      : L = L_CE
CE + Hidden  : L = L_CE + 0.1 L_hidden
CE + Align   : L = L_CE + 0.05 L_align
Full (Ours)  : L = L_CE + 0.1 L_hidden + 0.05 L_align
```

`L_align` is cosine alignment to the paired Lecture-CLAP text embedding. `L_hidden`
is MSE between the adapted hidden state and the detached original Whisper hidden state.

## 5. Parameter-matched PEFT comparison (RQ5-B)

Reported parameter-matched configurations:

```bash
# LoRA r=7, alpha=7, q/k/v/fc1/fc2 (~0.817M), reported LR 5e-4
METHOD=rq5b_lora_r7 GPU=1 bash command/run_mathspeech_reported_configs.sh

# Residual adapter bottleneck=128 (~0.790M), reported LR 5e-4
METHOD=rq5b_residual_b128 GPU=2 bash command/run_mathspeech_reported_configs.sh

# Ours (~0.790M), reported LR 3e-4
METHOD=ours GPU=3 bash command/run_mathspeech_reported_configs.sh
```

For the fixed-LR robustness comparison, override the two baselines to `3e-4`:

```bash
METHOD=rq5b_lora_r7 LR=3e-4 GPU=1 bash command/run_mathspeech_reported_configs.sh
METHOD=rq5b_residual_b128 LR=3e-4 GPU=2 bash command/run_mathspeech_reported_configs.sh
METHOD=ours LR=3e-4 GPU=3 bash command/run_mathspeech_reported_configs.sh
```

## 6. Representation analysis

The final mechanism analysis is implemented by:

```text
scripts/analyze_mathspeech_representation.py
scripts/plot_mathspeech_representation_mechanism.py
```

Run the analyzer separately for the `CE+Align` and `Full (Ours)` checkpoints on the
same validation manifest:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/analyze_mathspeech_representation.py \
  --manifest mathspeech/splits_source_balanced_seed42/mathspeech_val_source_seed42.csv \
  --ckpt /path/to/ce_align/best.pt \
  --clap_emb_path /path/to/mathspeech_clap_text_emb.pt \
  --output_dir /path/to/representation/ce_align

CUDA_VISIBLE_DEVICES=2 python scripts/analyze_mathspeech_representation.py \
  --manifest mathspeech/splits_source_balanced_seed42/mathspeech_val_source_seed42.csv \
  --ckpt /path/to/ours/best.pt \
  --clap_emb_path /path/to/mathspeech_clap_text_emb.pt \
  --output_dir /path/to/representation/full
```

Then generate the paper figure:

```bash
python scripts/plot_mathspeech_representation_mechanism.py \
  --ce_align_dir /path/to/representation/ce_align \
  --full_dir /path/to/representation/full \
  --output_dir /path/to/representation/mechanism
```

The analysis reports:

- paired semantic-alignment shift, `S_adapted - S_original`;
- hidden-state cosine preservation;
- hidden-state MSE;
- relative L2 representation drift;
- bootstrap 95% confidence intervals.

Semantic-alignment shifts should only be interpreted for variants whose AlignHead was
actually trained with the CLAP alignment objective. The analyzer emits a warning when
`lambda_align=0` or `align_loss_type=none`; for CE-only and CE+Hidden variants, use the
hidden-space preservation/drift metrics instead.

## 7. Key implementation details

The current Ours configuration uses:

```text
Backbone              : openai/whisper-base
Whisper parameters    : frozen
Adapter                : gated residual projector
Bottleneck             : 256
Pooling                : first Whisper encoder time step (`pool_type=cls`)
AlignHead              : LayerNorm -> Linear(hidden_dim, 512)
Adapter scale init     : 0.01
lambda_align           : 0.05
lambda_hidden          : 0.1
Seed                   : 42
Checkpoint selection   : best validation WER, beam=5
```

At inference time, Lecture-CLAP and AlignHead are not required for ASR generation; the
adapted Whisper encoder representation is passed directly to the frozen Whisper decoder.

## 8. Repository organization

```text
command/     current experiment launchers
mathspeech/  source-disjoint MathSpeech split manifests
scripts/     training, evaluation, collection, and analysis code
```

For MathSpeech, use `command/run_mathspeech_reported_configs.sh` as the single canonical
launcher. Obsolete MathSpeech grid, sweep, rerun, and parallel RQ launchers have been
removed from the active branch to avoid ambiguity. Their history remains available in Git.

S2L/M3AV scripts are retained as ongoing extension experiments and are not part of the
current finalized MathSpeech result table.
