# pylint: skip-file
# Pipeline functions
"""
    Pipelines for data manipulation
"""

# === Images ===

# -- Patches --
# Extract uniform patches from 2:3 photos
from .img_patches import batch_u23lp_sextract as impu23_extract

# -- Noise addition --
# Add gaussian noise
from .img_add_noise import add_gauss_noise_batch as imgs_add_awgn
from .img_add_noise import add_gauss_sigma_urange as \
    imgs_add_multinoise

# -- Shuffle --
# Image correspondences
from .shuffle_imgs import shuffle_img_img_corres as shuffle_corrimgs
