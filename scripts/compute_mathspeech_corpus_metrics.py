import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from jiwer import wer, cer
from transformers import WhisperProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--split_indices', required=True)
    p.add_argument('--split', default='test', choices=['train','valid','test'])
    p.add_argument('--pred_col', default=None)
    p.add_argument('--ref_col', default='transcription')
    p.add_argument('--model', default='openai/whisper-base')
    p.add_argument('--output_json', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    split = torch.load(args.split_indices, map_location='cpu', weights_only=False)
    indices = list(split[f'{args.split}_idx'])

    if args.pred_col is None:
        cols = [c for c in df.columns if c.startswith('pred_')]
        if not cols:
            raise ValueError(f'No pred_* column found: {df.columns.tolist()}')
        args.pred_col = cols[-1]

    processor = WhisperProcessor.from_pretrained(args.model, language='English', task='transcribe')
    tok = processor.tokenizer

    def norm(x):
        x = '' if pd.isna(x) else str(x).strip()
        if hasattr(tok, 'normalize'):
            return tok.normalize(x).strip()
        if hasattr(tok, '_normalize'):
            return tok._normalize(x).strip()
        raise RuntimeError('Whisper tokenizer normalizer not found')

    refs, hyps = [], []
    n_empty_pred = 0
    for i in indices:
        ref = norm(df.iloc[i][args.ref_col])
        hyp = norm(df.iloc[i][args.pred_col])
        if not ref:
            continue
        if not hyp:
            n_empty_pred += 1
        refs.append(ref)
        hyps.append(hyp)

    result = {
        'csv': args.csv,
        'split': args.split,
        'samples_in_split': len(indices),
        'valid_reference_samples': len(refs),
        'empty_predictions': n_empty_pred,
        'pred_col': args.pred_col,
        'wer': float(wer(refs, hyps)),
        'cer': float(cer(refs, hyps)),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
