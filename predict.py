"""Run a trained model on Landslide4Sense-format patches (128x128x14 .h5 files).

    python predict.py --model ResMUNet --threshold 0.5 --png datasets/images/test/image_1.h5
    python predict.py --model ResMUNet --threshold 0.5 datasets/images/test/   # whole folder

For each input ``<name>.h5`` writes ``<out-dir>/<name>_pred.h5`` holding
``prob`` (float32 landslide probability) and ``mask`` (uint8, prob >= threshold).
Use the threshold reported by ``evaluate.py`` / ``train.py`` for that checkpoint.
"""
import argparse
import os

import h5py
import numpy as np
import torch

from data import build_features, load_image, normalize
from models import MODELS, build_model, default_checkpoint


def collect_inputs(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files += sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5'))
        else:
            files.append(path)
    return files


def save_png(raw, prob, mask, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    feats = build_features(raw)
    rgb = np.moveaxis(feats[:3], 0, -1)
    rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)  # simple contrast stretch for display

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title, kw in [(axes[0], rgb, 'RGB', {}),
                               (axes[1], prob, 'Landslide probability', {'vmin': 0, 'vmax': 1}),
                               (axes[2], mask, 'Predicted mask', {'vmin': 0, 'vmax': 1, 'cmap': 'gray'})]:
        ax.imshow(img, **kw)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='.h5 patches or folders of them')
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--checkpoint', help='defaults to models/<model>_best_model.pth')
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--out-dir', default='predictions')
    p.add_argument('--png', action='store_true', help='also save an RGB / probability / mask figure')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint or default_checkpoint(args.model),
                                     map_location=device, weights_only=True))
    model.to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)

    files = collect_inputs(args.inputs)
    for path in files:
        raw = load_image(path)
        x = torch.from_numpy(normalize(build_features(raw))).float().unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        mask = (prob >= args.threshold).astype(np.uint8)

        stem = os.path.splitext(os.path.basename(path))[0]
        with h5py.File(os.path.join(args.out_dir, f'{stem}_pred.h5'), 'w') as f:
            f.create_dataset('prob', data=prob, compression='gzip')
            f.create_dataset('mask', data=mask, compression='gzip')
            f.attrs.update(model=args.model, threshold=args.threshold)
        if args.png:
            save_png(raw, prob, mask, os.path.join(args.out_dir, f'{stem}_pred.png'))

    print(f'Wrote {len(files)} prediction(s) to {args.out_dir}/')


if __name__ == '__main__':
    main()
