"""Train one model with one seed.

    python train.py --model ResMUNet --seed 0

Writes to ``runs/<model>/seed<seed>/``:
    best.pth       weights with the lowest validation loss
    history.json   per-epoch losses
    results.json   config, decision threshold chosen on the validation split,
                   validation metrics and (if the test split exists) test metrics

The released checkpoints in ``models/`` are never overwritten.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import LandslideDataset, split_dirs, train_val_split
from evaluator import evaluate, select_threshold
from losses import bce_dice_loss
from models import MODELS, build_model


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN determinism only; bilinear-upsampling backward on CUDA is still
        # non-deterministic, so runs are close but not bit-identical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--seed', type=int, default=0, help='initialisation / batch-order seed')
    p.add_argument('--split-seed', type=int, default=42, help='train/val split seed (keep fixed across runs)')
    p.add_argument('--data-root', default='datasets')
    p.add_argument('--out-dir', default='runs')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-patience', type=int, default=2)
    p.add_argument('--early-stop-patience', type=int, default=5)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--deterministic', action='store_true')
    return p.parse_args()


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            loss = bce_dice_loss(model(images), masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def main():
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_dir = os.path.join(args.out_dir, args.model, f'seed{args.seed}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, 'best.pth')

    train_set, val_set = train_val_split(LandslideDataset(*split_dirs(args.data_root, 'train')),
                                         seed=args.split_seed)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=args.num_workers, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor,
                                                     patience=args.lr_patience)

    print(f'{args.model} seed={args.seed} on {device}: {len(train_set)} train / {len(val_set)} val patches')
    history, best_val, best_epoch, stale = [], float('inf'), 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr})
        print(f'Epoch {epoch:3d}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  '
              f'lr {lr:.2e}  ({time.time() - t0:.0f}s)')

        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
            if stale >= args.early_stop_patience:
                print(f'Early stopping: no improvement for {stale} epochs.')
                break

    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Reload the best weights, choose the threshold on validation, then test once.
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    threshold, _ = select_threshold(model, val_loader, device)
    print(f'\nThreshold selected on validation: {threshold}')
    results = {
        'config': vars(args),
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'threshold': threshold,
        'val': evaluate(model, val_loader, device, threshold, verbose=False),
    }

    test_dirs = split_dirs(args.data_root, 'test')
    if all(os.path.isdir(d) for d in test_dirs):
        test_loader = DataLoader(LandslideDataset(*test_dirs), batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        print('\nTest split:')
        results['test'] = evaluate(model, test_loader, device, threshold)

    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {run_dir}')


if __name__ == '__main__':
    main()
