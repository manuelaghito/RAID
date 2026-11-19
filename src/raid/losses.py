import torch

## errors includes only sea cells.
## the average is computed by dividing a number of sea cells only 

def masked_mse(pred, target, mask):
    # pred, target: (B, C, H, W), one channel per target variable; mask: (B, 1, H, W) 1=sea, 0=land
    # mean over sea cells and channels (each variable is normalised, so they weigh equally)
    mask = mask.expand_as(pred)
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1)

