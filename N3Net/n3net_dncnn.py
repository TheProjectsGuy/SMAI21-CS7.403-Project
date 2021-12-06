# pylint: skip-file
# N3Net declaration with DnCNN
"""
    N3Net with DnCNN blocks as intermediate

    Creator: TheProjectsGuy
"""

# %% Import everything
# Core
import tensorflow as tf
from n3block import N3Block
from dncnn import DnCNN
from tensorflow.keras import Model

# %% Main class for the model
class N3Net(Model):
    def __init__(self, nc_in = 3, nc_out = 3, nb = 3, nc_i = 8, 
        residual = False, *args, **kwargs):
        """
            Creates an N3Net model using DnCNN and N3Block.

            Parameters:
            - nc_in: int   default: 3
                Number of input planes / channels. This is 1 for 
                grayscale and 3 for color images.
            - nc_out: int  default: 3
                Number of output channels (usually is the same as
                `nc_in`)
            - nb: int      default: 3
                Number of DnCNN blocks. The number of N3Blocks are 
                always 1 less than this, since they're put in between.
            - nc_i: int      default: 8
                Number of intermediate channels (for DnCNN output). 
                This is used for the input of N3Block (= output for
                DnCNN). The last DnCNN will have `nc_out` number of 
                output channels.
            - residual: bool    default: False
                Operation in residual mode. If True, then the final
                output (after running through intermediate stages) is 
                added with the original input and returned. The 
                addition is channel-wise.
        """
        super(N3Net, self).__init__(*args, **kwargs)
        # Parameters
        self.nb = nb    # Number of blocks (of DnCNN)
        self.residual = residual    # Residual connection
        self.nc_in = nc_in      # Number of input channels
        self.nc_out = nc_out    # Number of output channels
        # Models
        self.models_dncnn = []  # List of DnCNN, len = nb
        self.models_n3 = []     # List of N3Block, len = nb - 1
        # For the first pass
        self.models_dncnn.append(DnCNN(nc_in=nc_in, nc_out=nc_i, 
            name="dncnn_1"))
        self.models_n3.append(N3Block(name="nnn_1"))
        # For intermediate layers
        for i in range(1, nb-1):
            self.models_dncnn.append(DnCNN(nc_in=64, nc_out=nc_i, 
                name=f"dncnn_{i+1}"))
            self.models_n3.append(N3Block(name=f"nnn_{i+1}"))
        # For last layer (DnCNN)
        self.models_dncnn.append(DnCNN(nc_in=64, nc_out=nc_out, 
            name=f"dncnn_{nb}"))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        # Pass through the blocks
        for i in range(self.nb-1):
            x = self.models_dncnn[i](x)
            x = self.models_n3[i](x)
        # Final DnCNN
        x = self.models_dncnn[-1](x)    # Last element
        if self.residual:
            max_layers = min(self.nc_in, self.nc_out)
            x[:,:,:,:max_layers] += inputs[:,:,:,:max_layers]
        return x

# %% Experiment

# %%

# %%
