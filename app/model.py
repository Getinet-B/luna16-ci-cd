import torch
from monai.networks.nets import UNet


def build_unet() -> UNet:
    """
    Build the same 2D U-Net architecture used in the Colab project.
    """
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=1,
    )
    return model
