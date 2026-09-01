#!/bin/bash
set -euo pipefail

ROOT="/home/pej0918/Projects/Audio_Text"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path

LABEL_FIX_BLOCK = '''\
        # HF Whisper internally prepends decoder_start_token_id during training.
        # When the tokenizer has already inserted <|startoftranscript|> as the
        # first label token, remove it to avoid training the decoder to predict
        # a duplicate start token. This follows the standard Whisper collator
        # convention used in the team scripts.
        decoder_start_token_id = getattr(self.processor.tokenizer, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (
            labels.ndim == 2
            and labels.shape[1] > 0
            and decoder_start_token_id is not None
            and decoder_start_token_id >= 0
            and (labels[:, 0] == decoder_start_token_id).all().item()
        ):
            labels = labels[:, 1:]
'''


def patch_once(path, old, new, marker):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = p.read_text()
    if marker in text:
        print(f"[SKIP already patched] {path}")
        return False
    if old not in text:
        raise RuntimeError(f"Pattern not found in {path}:\n{old}")
    p.write_text(text.replace(old, new, 1))
    print(f"[PATCHED] {path}")
    return True


# 1) Generic ASR collator: used by LoRA, KAUST-style residual, and S2L generic training.
old = '''\
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        return {
'''
new = '''\
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
''' + LABEL_FIX_BLOCK + '''\
        return {
'''
patch_once(
    "asr_dataset_utils.py",
    old,
    new,
    "HF Whisper internally prepends decoder_start_token_id during training",
)

# 2) Full fine-tuning collator.
old = '''\
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        real_idx = torch.tensor([x["real_idx"] for x in batch], dtype=torch.long)
'''
new = '''\
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
''' + LABEL_FIX_BLOCK + '''\
        real_idx = torch.tensor([x["real_idx"] for x in batch], dtype=torch.long)
'''
patch_once(
    "train_whisper_ft.py",
    old,
    new,
    "HF Whisper internally prepends decoder_start_token_id during training",
)

# 3) CLAP adapter collator: used by Ours and CLAP-guided Full FT.
old = '''\
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        output = {
'''
new = '''\
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
''' + LABEL_FIX_BLOCK + '''\

        output = {
'''
patch_once(
    "train_whisper_clap_adapter.py",
    old,
    new,
    "HF Whisper internally prepends decoder_start_token_id during training",
)

# 4) Match LoRA-Whisper-style target set more closely by including out_proj.
# This affects the generic train script default.
p = Path("train_whisper_lora.py")
text = p.read_text()
if 'default="q_proj,k_proj,v_proj,fc1,fc2"' in text:
    text = text.replace(
        'default="q_proj,k_proj,v_proj,fc1,fc2"',
        'default="q_proj,k_proj,v_proj,out_proj,fc1,fc2"',
        1,
    )
    p.write_text(text)
    print("[PATCHED] train_whisper_lora.py target_modules default adds out_proj")
elif 'default="q_proj,k_proj,v_proj,out_proj,fc1,fc2"' in text:
    print("[SKIP already patched] train_whisper_lora.py target_modules default")
else:
    print("[WARN] train_whisper_lora.py target_modules default pattern not found; check manually")

# 5) Also update the controlled MathSpeech LoRA runner so the explicit CLI target set includes out_proj.
p = Path("bash/run_mathspeech_lora_whisper_controlled_epoch10_beam5.sh")
if p.exists():
    text = p.read_text()
    if '--target_modules q_proj,k_proj,v_proj,fc1,fc2' in text:
        text = text.replace(
            '--target_modules q_proj,k_proj,v_proj,fc1,fc2',
            '--target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2',
            1,
        )
        p.write_text(text)
        print("[PATCHED] controlled LoRA runner target_modules adds out_proj")
    elif '--target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2' in text:
        print("[SKIP already patched] controlled LoRA runner target_modules")
    else:
        print("[WARN] controlled LoRA runner target_modules pattern not found; check manually")

print("\n[DONE] Whisper label-shift fix applied.")
print("Verify with:")
print('  grep -R "decoder_start_token_id\|startoftranscript" -n asr_dataset_utils.py train_whisper_ft.py train_whisper_clap_adapter.py')
print('  grep -R "target_modules" -n train_whisper_lora.py bash/run_mathspeech_lora_whisper_controlled_epoch10_beam5.sh')
PY
