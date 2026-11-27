from .cnn import SimpleSRCNN
from .unet import UNet



def build_model(cfg):
    m = cfg.model

    ## input channels follow the data config, so adding a static field can't silently mismatch
    in_channels = len(cfg.data.input_vars) + len(cfg.data.static_inputs)
    ## one output per target variable, each residual over the input channel of the same variable:
    ## the models add their correction to x[:, :n_out], so input_vars must start with target_vars
    targets = list(cfg.data.target_vars)
    n_out = len(targets)
    if list(cfg.data.input_vars[:n_out]) != targets:
        raise ValueError(f"input_vars {cfg.data.input_vars} must start with target_vars {targets} "
                         "(output k is a correction to input channel k)")
    if m.name == "cnn":
        return SimpleSRCNN(in_channels, getattr(m, "hidden_dim", 64), getattr(m, "n_blocks", 4), n_out)
    if m.name == "unet":
        return UNet(in_channels, getattr(m, "base_channels", 32), getattr(m, "depth", 3), n_out)

    raise ValueError(f"Unknown model: {m.name}")