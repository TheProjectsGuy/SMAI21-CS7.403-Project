#dev: Nayan
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


    def embedding_block(self, inp_s):
        input_shape=inp_s
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3,3),padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, kernel_size=(3,3),padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(8, kernel_size=(3,3),padding="same", input_shape=input_shape))
        return model


    def temperature_parameter(self, inp_s):
        input_shape=inp_s
        model2 = Sequential()
        model2.add(Conv2D(64, kernel_size=(3,3),padding="same", input_shape=input_shape))
        model2.add(BatchNormalization())
        model2.add(Activation('relu'))
        model2.add(Conv2D(64, kernel_size=(3,3),padding="same", input_shape=input_shape))
        model2.add(BatchNormalization())
        model2.add(Activation('relu'))
        model2.add(Conv2D(3, kernel_size=(3,3),padding="same", input_shape=input_shape))
        return model2

# %%
