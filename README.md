# Landslide

UNet, ResUNet, MUNet and ResMUNet for pixel-wise landslide segmentation on the
[Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022) benchmark.
Inputs are 6 channels built from each 14-band patch: Red, Green, Blue, NDVI, Slope, Elevation.

## Setup

```bash
pip install -r requirements.txt
```

Put the data under `datasets/` so that images `image_N.h5` and masks `mask_N.h5` live in:

```
datasets/images/{train,test}/
datasets/annotations/{train,test}/
```

## Code layout

| File | Purpose |
|---|---|
| `models/unet_family.py` | All four architectures (two switches: `residual`, `multiscale`). Models return logits. |
| `data.py` | Feature construction, normalisation, dataset, the fixed 80/20 train/val split |
| `losses.py` | BCE + Dice loss on logits |
| `evaluator.py` | Confusion-matrix metrics and validation-based threshold selection |
| `train.py` | Train one model with one seed |
| `evaluate.py` | Evaluate a checkpoint on the test split |
| `predict.py` | Run a checkpoint on `.h5` patches |
| `summarize.py` | Mean ± std over seeds, Welch t-test between two models |
| `scripts/run_experiments.sh` | All models × seeds 0–4, then the summary |

## Usage

```bash
# Evaluate a released checkpoint (models/<Model>_best_model.pth);
# the decision threshold is chosen on the validation split, not on test.
python evaluate.py --model ResMUNet

# Train one run -> runs/ResMUNet/seed0/{best.pth,history.json,results.json}
python train.py --model ResMUNet --seed 0

# Full multi-seed comparison
bash scripts/run_experiments.sh
python summarize.py --compare ResMUNet ResUNet

# Predict on new patches
python predict.py --model ResMUNet --threshold <from evaluate.py> --png datasets/images/test/
```

## Metrics

A pixel is labelled landslide when `sigmoid(logit) >= threshold`. Metrics are
computed over all pixels of the split. `iou_landslide` is the IoU of the
landslide class; `miou` averages the landslide and background IoU. Overall
accuracy is reported but is uninformative here (~98% of pixels are background).

The numbers in the accompanying PDF came from an earlier evaluator that applied
the sigmoid twice. Its `threshold=0.6` therefore meant a probability threshold
of about 0.405, and that threshold was not chosen on validation data. Re-run the
scripts above to get numbers produced by the current pipeline.
