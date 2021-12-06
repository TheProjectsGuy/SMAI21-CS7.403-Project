# pylint: skip-file
# DnCNN Block
"""
    DnCNN block for the intermediate and final layers

    Current Worker: TheProjectsGuy
"""

# %% Import everything
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU

# %%
class DnCNN(Model):
    def __init__(self, nc_in = 3, nc_out = 8, ndepth=6, *args, **kwargs):
        """
            Creates and returns a DnCNN model.

            Parameters:
            - ndepth: int       default: 6
                The number of layers to use for DnCNN model. Note that
                `ndepth-1` layers are (Conv + BN + ReLU) and the final
                layer is just Conv.
            - nc_in: int        default: 3
                The number of input channels for the model
            - nc_out: int       default: 8
                The number of output channels for the model
        """
        super(DnCNN, self).__init__(*args, **kwargs)
        # Parameters
        self.ndepth: int = ndepth
    pass
