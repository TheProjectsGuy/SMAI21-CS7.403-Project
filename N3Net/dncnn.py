# pylint: skip-file
# DnCNN Block
"""
    DnCNN block for the intermediate and final layers

    Creator: TheProjectsGuy
"""

# %% Import everything
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU

# %%
class DnCNN(Model):
    def __init__(self, s=80, nk=64, ks=(3,3), nc_in = 3, nc_out = 8, 
        nd=6, residual = True, *args, **kwargs):
        """
            Creates and returns a DnCNN model.

            Parameters:
            - s: int            default: 80
                The size of the square image (height and width)
            - nk: int           default: 64
                The number of kernels to use (for the convolutions).
                This is for the input and the intermediate layers. The
                output layer will use the `nc_out` parameter.
            - ks: Tuple(int, int)   default: (3, 3)
                The kernel size, (height, width) for each kernel
            - nc_in: int        default: 3
                The number of input channels for the model (for the
                first layer to accept).
            - nc_out: int       default: 8
                The number of output channels for the model (for the
                last layer to output).
            - nd: int       default: 6
                The number of layers to use for DnCNN model. Note that
                `nd-1` layers are (Conv + BN + ReLU) and the final
                layer is just (Conv).
            - residual: bool    default: True
                If true, then the residual connection is used. Here,
                the output of layers is added to the original input
                and then returned. Only the maximum number of possible
                layers are added (for the skip connection); the least
                among (`nc_out` and `nc_in`).
        """
        super(DnCNN, self).__init__(*args, **kwargs)
        # Parameters
        self.residual = residual    # Residual, for call
        self.nc_out = nc_out    # Number of output channels
        self.nc_in = nc_in      # Number of input channels
        # Model
        model = models.Sequential(name=f"{self.name}_seq")
        # All the Conv + BN + ReLU layers
        model.add(Conv2D(nk, ks, padding='same', 
            input_shape=(s, s, nc_in)))
        model.add(BatchNormalization())
        model.add(ReLU())
        for _ in range(1, nd-1):
            model.add(Conv2D(nk, ks, padding='same'))
            model.add(BatchNormalization())
            model.add(ReLU())
        # Final convolution layer
        model.add(Conv2D(nc_out, ks, padding='same'))
        self.seq_model = model  # Sequential model (no residual)

    # Call function
    def call(self, inputs, training=None, mask=None):
        # Forward pass
        x = inputs
        x = self.seq_model(x, training, mask)   # Encapsulated
        if self.residual:
            max_layers = min(self.nc_in, self.nc_out)
            x[:,:,:,:max_layers] += inputs[:,:,:,:max_layers]
        return x

# %%
