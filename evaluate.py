"""Evaluate a checkpoint on the test split.

    # released checkpoint, threshold tuned on the validation split (default)
    python evaluate.py --model ResMUNet

    # your own run, fixed threshold
    python evaluate.py --model ResMUNet --checkpoint runs/ResMUNet/seed0/best.pth --threshold 0.5

The threshold is never tuned on the split being evaluated.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from data import LandslideDataset, split_dirs, train_val_split
from evaluator import evaluate, select_threshold
from models import MODELS, build_model, default_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--checkpoint', help='defaults to models/<model>_best_model.pth')
    p.add_argument('--threshold', default='auto',
                   help="probability threshold, or 'auto' to pick the F1-optimal one on the validation split")
    p.add_argument('--data-root', default='datasets')
    p.add_argument('--split', default='test')
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--out', help='write metrics to this JSON file')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = args.checkpoint or default_checkpoint(args.model)
    model = build_model(args.model)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    print(f'{args.model} <- {ckpt}')

    if args.threshold == 'auto':
        _, val_set = train_val_split(LandslideDataset(*split_dirs(args.data_root, 'train')), seed=args.split_seed)
        threshold, val_f1 = select_threshold(model, DataLoader(val_set, batch_size=args.batch_size), device)
        print(f'Threshold selected on validation split: {threshold} (val F1 {val_f1:.4f})')
    else:
        threshold = float(args.threshold)

    loader = DataLoader(LandslideDataset(*split_dirs(args.data_root, args.split)),
                        batch_size=args.batch_size, shuffle=False)
    print(f'\n{args.split} split:')
    metrics = evaluate(model, loader, device, threshold)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'model': args.model, 'checkpoint': ckpt, 'split': args.split, **metrics}, f, indent=2)


if __name__ == '__main__':
    main()
