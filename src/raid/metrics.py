import numpy as np

## Calculations are performed using values converted back to physical units
## (allowing for explanations such as "RMSE of 0.3 degreeC").
## The same function is used to evaluate both the bicubic baseline and the CNN/U-Net models.

## physical units of the fields, for labels; unknown names print without a unit
UNITS = {"temp": "degC", "salt": "psu", "u": "m/s", "v": "m/s", "ubar": "m/s", "vbar": "m/s", "zeta": "m"}

def masked_rmse(pred, target, mask):
    m = mask.astype(bool)
    return np.sqrt(((pred - target)[..., m] ** 2).mean())

def masked_bias(pred, target, mask):
    m = mask.astype(bool)
    return (pred - target)[..., m].mean()


