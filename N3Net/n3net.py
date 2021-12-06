# pylint: skip-file
# N3Net declaration with DnCNN
"""
    N3Net with DnCNN blocks as intermediate

    Current Worker: Nobody

    Kept for implementing in the end
"""

# %% Import everything
# Core
import tensorflow as tf
from n3block import N3Block
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU

# %% Main class for the model
class N3Net(Model):
    def __init__(self, nplanes_in = 3, nplanes_out = 3, nblocks = 3, nplanes_interim = 8, residual = False, *args, **kwargs):
        """
            Creates an N3Net model using DnCNN and N3Block.

            Parameters:
            - nplanes_in: int   default: 3
                Number of input planes
            - nplanes_out: int  default: 3
                Number of output planes
            - nblocks: int      default: 3
                Number of DnCNN blocks. The number of N3Blocks are 
                always 1 less than this, since they're put in between
            - nplanes_interim: int      default: 8
                Number of intermediate planes (for DnCNN output). This
                is used for the input of N3Block as well as for the
                output for DnCNN. The last DnCNN will have 
                `nplanes_out` number of output planes.
            - residual: bool    default: False
                Operating in residual mode. If True, then the final
                output (after running through intermediate stages) is 
                added with the original input and returned. The 
                addition is channel-wise.
        """
        super(Model, self).__init__(*args, **kwargs)
        # Parameters
        self.nplanes_in: int = nplanes_in
        self.nplanes_out: int = nplanes_out
        self.nblocks: int = nblocks
        self.residual: bool = residual
        self.nplanes_interim: int = nplanes_interim
        # Local parameters
        self.cnn_models = []    # len = self.nblocks
        self.n3b_models = []    # len = self.nblocks - 1
        # DnCNN -> N3Block = single block
        for i in range(self.nblocks):
            # DnCNN model (3 channels, image denoising)
            n_out = self.nplanes_out if i == self.nblocks - 1 \
                else self.nplanes_interim
            dn_cnn = models.Sequential(name=f"dncnn-{i}", layers=[
                # Conv/BN/ReLU
                Conv2D(64, (3, 3), padding='same',
                    input_shape=(80, 80, 3)),
                BatchNormalization(),
                ReLU(),
                # Conv/BN/ReLU
                Conv2D(64, (3, 3), padding='same'),
                BatchNormalization(),
                ReLU(),
                # Conv/BN/ReLU
                Conv2D(64, (3, 3), padding='same'),
                BatchNormalization(),
                ReLU(),
                # Conv/BN/ReLU
                Conv2D(64, (3, 3), padding='same'),
                BatchNormalization(),
                ReLU(),
                # Conv/BN/ReLU
                Conv2D(64, (3, 3), padding='same'),
                BatchNormalization(),
                ReLU(),
                # Conv
                Conv2D(n_out, (3, 3), padding='same',
                    input_shape=(80, 80, 3))
            ])
            self.cnn_models.append(dn_cnn)
            if i < self.nblocks - 1:
                # N3Block
                nnn_block = N3Block()
                self.n3b_models.append(nnn_block)
            
    def call(self, inputs, training=None, mask=None):
        # Forward pass of N3Net
        x = inputs
        in_x = x    # Backup of inputs
        for i in range(self.nblocks-1):
            x = self.cnn_models[i](x)   # DnCNN
            x = self.n3b_models[i](x)   # N3Block
        # Final layer
        x = self.cnn_models[self.nblocks-1](x)  # Final DnCNN
        # If residual

        return x

# %% Experiment

# %%


# %%
