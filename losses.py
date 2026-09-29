import torch
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1e-6):
    """Soft Dice loss per image, averaged over the batch."""
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()


def bce_dice_loss(logits, target):
    """BCE + Dice on raw logits (BCE-with-logits is the numerically stable form)."""
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)
