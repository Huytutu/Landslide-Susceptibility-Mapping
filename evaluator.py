"""Pixel-level metrics for binary landslide segmentation.

Models return logits; sigmoid is applied here, once. A pixel is predicted as
landslide when ``probability >= threshold``.

Metrics are computed from the confusion matrix accumulated over the whole
split (micro-averaged over pixels), so nothing is ever flattened into memory.
"""
import numpy as np
import torch

# probability histogram resolution used for threshold selection
_N_BINS = 1000


def metrics_from_confusion(tn, fp, fn, tp):
    tn, fp, fn, tp = (int(v) for v in (tn, fp, fn, tp))
    n = tn + fp + fn + tp

    def div(a, b):
        return a / b if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    accuracy = div(tp + tn, n)
    p_e = div((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn), n * n)
    iou_landslide = div(tp, tp + fp + fn)
    iou_background = div(tn, tn + fn + fp)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': div(2 * precision * recall, precision + recall),
        'kappa': div(accuracy - p_e, 1 - p_e),
        'iou_landslide': iou_landslide,             # IoU of the landslide class (what the paper reported as "Mean IoU")
        'iou_background': iou_background,
        'miou': (iou_landslide + iou_background) / 2,  # mean over both classes
        'confusion_matrix': {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp},
    }


@torch.no_grad()
def _iterate_probs(model, dataloader, device):
    model.eval()
    for images, masks in dataloader:
        probs = torch.sigmoid(model(images.to(device)))
        yield probs, masks.to(device) > 0.5


@torch.no_grad()
def confusion(model, dataloader, device, threshold):
    tn = fp = fn = tp = 0
    for probs, labels in _iterate_probs(model, dataloader, device):
        preds = probs >= threshold
        tp += (preds & labels).sum().item()
        fp += (preds & ~labels).sum().item()
        fn += (~preds & labels).sum().item()
        tn += (~preds & ~labels).sum().item()
    return tn, fp, fn, tp


@torch.no_grad()
def select_threshold(model, dataloader, device, grid=np.round(np.arange(0.05, 0.951, 0.01), 2)):
    """Pick the probability threshold that maximises F1 on ``dataloader``.

    Use a validation split for this, never the test split.
    Returns ``(best_threshold, f1_at_best)``.
    """
    pos_hist = torch.zeros(_N_BINS, dtype=torch.float64, device=device)
    neg_hist = torch.zeros(_N_BINS, dtype=torch.float64, device=device)
    for probs, labels in _iterate_probs(model, dataloader, device):
        bins = (probs * _N_BINS).long().clamp_(0, _N_BINS - 1)
        pos_hist += torch.bincount(bins[labels], minlength=_N_BINS).double()
        neg_hist += torch.bincount(bins[~labels], minlength=_N_BINS).double()

    # counts of pixels with prob >= k / _N_BINS
    tp_at = pos_hist.flip(0).cumsum(0).flip(0).cpu().numpy()
    fp_at = neg_hist.flip(0).cumsum(0).flip(0).cpu().numpy()
    n_pos = pos_hist.sum().item()

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        k = int(round(t * _N_BINS))
        tp, fp = tp_at[k], fp_at[k]
        f1 = 2 * tp / (2 * tp + fp + (n_pos - tp)) if tp else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def format_metrics(m):
    cm = m['confusion_matrix']
    return '\n'.join([
        f"F1-Score:           {m['f1_score']:.4f}",
        f"Precision:          {m['precision']:.4f}",
        f"Recall:             {m['recall']:.4f}",
        f"IoU (landslide):    {m['iou_landslide']:.4f}",
        f"mIoU (2 classes):   {m['miou']:.4f}",
        f"Kappa Coefficient:  {m['kappa']:.4f}",
        f"Overall Accuracy:   {m['accuracy']:.4f}  (dominated by background; not informative)",
        '',
        'Confusion Matrix:',
        f"TN: {cm['TN']} | FP: {cm['FP']}",
        f"FN: {cm['FN']} | TP: {cm['TP']}",
    ])


def evaluate(model, dataloader, device, threshold=0.5, verbose=True):
    """Metrics of ``model`` on ``dataloader`` at a fixed probability threshold."""
    m = metrics_from_confusion(*confusion(model, dataloader, device, threshold))
    m['threshold'] = threshold
    if verbose:
        print(f'Threshold:          {threshold}')
        print(format_metrics(m))
    return m
