# pylint: skip-file
# First Model of DnCNN trained using simple TensorFlow
"""
    Currently, the module shouldn't be imported as it's not ready.
    RUN ONLY AS MAIN
    
    Creator: TheProjectsGuy
"""

if __name__ != "__main__":
    raise ImportError("This module shouldn't be imported")

# %% Path gimmick
# This is done to add the entire repository as if it's a module
import os
import sys
from pathlib import Path
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARN: __file__ not found, trying local")
    dir_name = os.path.abspath('')
# Top project directory (for accessing everything)
pkg_path = str(Path(dir_name).parent.parent.parent)
# Add to path
if pkg_path not in sys.path:
    sys.path.append(pkg_path)
    print("Added package path")

# %% Import everything
# All necessary libraries
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# Keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
# -- Project Modules --
# BSDS500 dataset
from DataLoaders.BSDS500 import load_data, load_images
# Image operations
from Pipelines import imgs_add_multinoise as add_noise
from Pipelines import impu23_extract as extract_patches
from Pipelines import shuffle_corrimgs as shuffle_imgs

# %% System verification
# TensorFlow
print(f"Tensorflow Version: {tf.__version__}")
devs = tf.config.list_physical_devices()
for dev in devs:
    print(f"Found device: {dev}")
    if dev.device_type == "GPU":
        print("CUDA acceleration available")

# %% Dataset
# Load BSDS500 into memory
bsds500_data = load_data()
# Only images
train432_images, test68_images = load_images(bsds500_data)

# %% Configurations
# -- User parameters --
# Paraemters
patch_size = 50 # Patches to use (default: 50)
num_batches = 16   # Number of batches (128)
ns_pbat = 500  # Number of samples per batch (3000)
sig_range = [0, 55] # Range of sigma values
sig_nums = 17   # Number of samples U[sig_range] for noise
# -- Model parameters --
cnn_depth = 20  # Depth of model
# Checkpointing
ckpt_path = "./checkpoints/24_nov_21/cp_{epoch}.ckpt"
ckpt_dir = os.path.dirname(ckpt_path)
# -- Other parameters: Change only if you know it! --
stride_c = [53, 54] # Major, Minor strides (patch generation)

# %% Prepare training data
# Add noise
print("Adding noise")
noisy_imgs, clean_imgs = add_noise(train432_images, sig_range[0], 
    sig_range[1], sig_nums, 0)
print(f"Added noise to {len(noisy_imgs)} images")
print("Extracting patches")
# Extract patches
noisy_ps = extract_patches(noisy_imgs, patch_size, stride_c)
clean_ps = extract_patches(clean_imgs, patch_size, stride_c)
print(f"Extracted {len(noisy_ps)} patches of {noisy_ps[0].shape}")
# Shuffle indices and convert to floating point for training
x_imgs, y_imgs = shuffle_imgs(noisy_ps, clean_ps, num_batches*ns_pbat)
x_fimgs = np.array(x_imgs, float)   # Noisy: Inputs
y_fimgs = np.array(y_imgs, float)   # Clean = noisy - residuals
# Compute residuals (noise = clean + residual)
r_fimgs = x_fimgs - y_fimgs # These are expected outputs
print(f"Have {x_fimgs.shape[0]} images to train")

# %% Create and compile model
# Main sequential model
model = models.Sequential(name="CDnCNN-B")
# First layer
model.add(layers.Conv2D(64, (3, 3), activation=activations.relu,
    padding='same', input_shape=(50, 50, 3), name="crlu_1"))
# 2nd to D-1th layer
for i in range(2, cnn_depth):   # 2 to depth
    # Add Conv + Batch Norm + Relu
    model.add(layers.Conv2D(64, (3, 3), padding='same', 
        name=f"conv_{i}"))
    model.add(layers.BatchNormalization(name=f"bn_{i}"))
    model.add(layers.Activation(activations.relu, name=f"relu_{i}"))
# For the last stage
model.add(layers.Conv2D(3, (3, 3), padding='same', name="conv_o"))
print("Model creation successful")
model.summary()
# Compile the model
model.compile(optimizer=optimizers.Adam(),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()])
# Save model
model.save(f"{ckpt_dir}/model")
print("Model saved")

# %% Fit the model
# Checkpoint training
cp_callback = callbacks.ModelCheckpoint(ckpt_path, verbose=1, 
    save_weights_only=True)
# Train to find residuals for given noisy image
history = model.fit(x_fimgs, r_fimgs, batch_size=num_batches, 
    epochs=5, verbose=2, callbacks=[cp_callback])

# %%

# %%

# %% Experimental section
# The following cells are for experiment only

# %% Plot
i = np.random.randint(0, len(x_imgs))
plt.figure(figsize=(10, 10))
plt.subplot(1,2,1)
plt.imshow(x_imgs[i])
plt.subplot(1,2,2)
plt.imshow(y_imgs[i])
mse_loss = losses.MeanSquaredError()
mse_loss(y_fimgs[i], x_fimgs[i]).numpy()

# %% Test performance
plt.figure(figsize=(10, 10))
plt.subplot(1,2,1)
plt.imshow(x_imgs[i])
plt.subplot(1,2,2)
res_pred = model(x_fimgs[[i]]).numpy()[0]
y_pred = x_fimgs[i] - res_pred
y_predf = (255*(y_pred - np.min(y_pred))/np.ptp(y_pred))
y_pred = y_predf.astype(np.uint8)
plt.imshow(y_pred)
mse_loss = losses.MeanSquaredError()
mse_loss(y_fimgs[i], y_predf).numpy()

# %% Load model
model_path = f"{ckpt_dir}/model"
# Loaded model
_model_loaded = models.load_model(model_path)
# Load weights
_model_loaded.load_weights(ckpt_path.format(epoch=5))
print("Model loaded again")

# %% Test performance (on loaded model)
plt.figure(figsize=(10, 10))
plt.subplot(1,2,1)
plt.imshow(x_imgs[i])
plt.subplot(1,2,2)
_res_pred = _model_loaded(x_fimgs[[i]]).numpy()[0]
_y_pred = x_fimgs[i] - _res_pred
_y_predf = (255*(_y_pred - np.min(_y_pred))/np.ptp(_y_pred))
_y_pred = _y_predf.astype(np.uint8)
plt.imshow(_y_pred)
if np.allclose(y_predf, _y_predf):
    print("The model was saved correctly")

# %%
