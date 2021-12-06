# pylint: skip-file
# N3Net declaration with DnCNN
"""
    N3Net with DnCNN blocks as intermediate

    Current Worker: TheProjectsGuy
"""

# %% Import everything
# Core
import tensorflow as tf
from n3block import N3Block
from dncnn import DnCNN
from tensorflow.keras import Model

# %% Main class for the model
class N3Net(Model):
    def __init__(self, nc_in = 3, *args, **kwargs):
        """
            Creates an N3Net model using DnCNN and N3Block.

            Parameters:
            - nc_in: int   default: 3
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

            
    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training=training, mask=mask)

# %% Experiment

# %%


# %%
