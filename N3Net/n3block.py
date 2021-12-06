# pylint: skip-file
# N3Block
"""
    Neural Nearest Neighbor block implementation
"""

# %% Import everything
import tensorflow as tf
from tensorflow.keras import Model

# %% Classes
# Implements Neural Nearest Neighbors
class N3Block(Model):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

# %%
