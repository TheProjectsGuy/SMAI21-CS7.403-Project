# pylint: skip-file
# N3Block
"""
    Neural Nearest Neighbor block implementation

    Log:
    
    - Creator: nayanjha16
        Created the file

    - Modified: TheProjectsGuy
        Modified imports for functionality
"""

# %% Import everything
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    Activation

# %% Classes
# Implements Neural Nearest Neighbors
class N3Block(Model):
    @staticmethod
    def embedding_block(inp_s):
        input_shape=inp_s
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3,3),padding="same",
            input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, kernel_size=(3,3),padding="same", 
            input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(8, kernel_size=(3,3),padding="same", 
            input_shape=input_shape))
        return model

    @staticmethod
    def temperature_parameter(inp_s):
        input_shape=inp_s
        model2 = Sequential()
        model2.add(Conv2D(64, kernel_size=(3,3),padding="same", 
            input_shape=input_shape))
        model2.add(BatchNormalization())
        model2.add(Activation('relu'))
        model2.add(Conv2D(64, kernel_size=(3,3),padding="same", 
            input_shape=input_shape))
        model2.add(BatchNormalization())
        model2.add(Activation('relu'))
        model2.add(Conv2D(3, kernel_size=(3,3),padding="same", 
            input_shape=input_shape))
        return model2
    
    def __init__(self, nc_in = 8, k = 7, *args, **kwargs):
        """
            Creates an N3Block with Neural Nearest Neighbor 
            implementation.

            Parameters:
            - nc_in: int        default: 8
                Number of input channels
            - k: int            default: 7
                Number of neighbors to find
        """
        super(N3Block, self).__init__(*args, **kwargs)

# %%
