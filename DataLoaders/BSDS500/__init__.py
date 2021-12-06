# pylint: skip-file
# BSDS500 Dataset Loader
"""
    All handlers for BSDS500 dataset

    Creator: TheProjectsGuy
"""

# Main dataset and class
from . import BSDS500
from .BSDS500 import BSDS500_DataSet

# -- Helper functions --
# Load entire dataset
from .def_load_dataset import load_bsds500_dataset as load_data
# Load only the images (split provision)
from .def_load_dataset import load_imgs_dataset as load_images
