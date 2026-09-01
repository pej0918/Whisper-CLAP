#!/bin/bash
set -euo pipefail

ROOT="/home/pej0918/Projects/Audio_Text"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path

LABEL_FIX_BLOCK = '''\
        # HF Whisper internally prepends decoder_start_token_id during training.
        # If the tokenizer already inserted <|startoftranscript|> as the first
        # label token, remove it so the model is not trained to predict a
        # duplicate start token. This matches the team Whisper collator setup.
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

MARKER = 'HF Whisper internally prepends decoder_start_token_id during training'


def _next_class_pos(text: str, start: int) -> int:
    pos = text.find('\nclass ', start + 1)
    return len(text) if pos < 0 else pos


def patch_class_before(path: str, class_name: str, needle: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = p.read_text()
    start = text.find(f'class {class_name}')
    if start < 0:
        raise RuntimeError(f'class {class_name} not found in {path}')
    end = _next_class_pos(text, start)
    block = text[start:end]
    if MARKER in block:
        print(f'[SKIP already patched] {path}::{class_name}')
        return False
    rel = block.find(needle)
    if rel < 0:
        raise RuntimeError(
            f'needle not found in {path}::{class_name}: {needle!r}\n'
            f'Inspect around the collator and insert the label fix manually.'
        )
    insert_at = start + rel
    text = text[:insert_at] + LABEL_FIX_BLOCK + text[insert_at:]
    p.write_text(text)
    print(f'[PATCHED] {path}::{class_name}')
    return True


# 1) Generic ASR collator: LoRA / KAUST-style residual / S2L generic training.
patch_class_before(
    'asr_dataset_utils.py',
    'ASRCollator',
    '        return {',
)

# 2) Full fine-tuning collator.
patch_class_before(
    'train_whisper_ft.py',
    'WhisperCollator',
    '        real_idx = torch.tensor([x["real_idx"] for x in batch], dtype=torch.long)',
)

# 3) CLAP adapter collator: Ours / CLAP-guided Full FT.
patch_class_before(
    'train_whisper_clap_adapter.py',
    'DataCollatorSpeechSeq2SeqWithClap',
    '        output = {',
)

# 4) Keep LoRA-Whisper target set aligned with the paper: Wq, Wk, Wv, and FC.
# In Hugging Face Whisper this maps to q_proj, k_proj, v_proj, fc1, fc2.
p = Path('train_whisper_lora.py')
text = p.read_text()
if 'default="q_proj,k_proj,v_proj,out_proj,fc1,fc2"' in text:
    text = text.replace(
        'default="q_proj,k_proj,v_proj,out_proj,fc1,fc2"',
        'default="q_proj,k_proj,v_proj,fc1,fc2"',
        1,
    )
    p.write_text(text)
    print('[PATCHED] train_whisper_lora.py target_modules removes out_proj')
elif 'default="q_proj,k_proj,v_proj,fc1,fc2"' in text:
    print('[OK] train_whisper_lora.py target_modules already paper-aligned')
else:
    print('[WARN] train_whisper_lora.py target_modules default pattern not found; check manually')

# 5) Controlled MathSpeech LoRA runner uses explicit target modules; keep it paper-aligned too.
p = Path('bash/run_mathspeech_lora_whisper_controlled_epoch10_beam5.sh')
if p.exists():
    text = p.read_text()
    if '--target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2' in text:
        text = text.replace(
            '--target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2',
            '--target_modules q_proj,k_proj,v_proj,fc1,fc2',
            1,
        )
        p.write_text(text)
        print('[PATCHED] controlled LoRA runner target_modules removes out_proj')
    elif '--target_modules q_proj,k_proj,v_proj,fc1,fc2' in text:
        print('[OK] controlled LoRA runner target_modules already paper-aligned')
    else:
        print('[WARN] controlled LoRA runner target_modules pattern not found; check manually')

print('\n[DONE] Whisper label-shift fix applied.')
print('Verify with:')
print('  grep -R "decoder_start_token_id\\|startoftranscript" -n asr_dataset_utils.py train_whisper_ft.py train_whisper_clap_adapter.py')
print('  grep -R "target_modules" -n train_whisper_lora.py bash/run_mathspeech_lora_whisper_controlled_epoch10_beam5.sh')
PY
