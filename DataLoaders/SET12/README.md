# SET12 DATASET

SET12  is a dataset primarily for Model evaluation and comparison.

**Home Page**:You can find it on [LINK](https://github.com/aGIToz/KerasDnCNN/tree/master/Set12).

## About the dataset
SET12 DATASET CONTAIN SET OF 12 IMAGES.


1. **Images**: There are 12 images in the dataset. Each image is either `321, 481` or `481, 321` (represented as `height, width`) and in channel RGB (8-bit color). These are stored as PNG files.

### Functions

The module has the following functions that can be used accordingly to load the dataset

 `load_data`: Loads the entire SET12 dataset (images, paths, everything related to the dataset). The header of this function is described below
    ```py
    def train_test_split_for_image_SET12():
    """
        This function returns the images from SET12 folder
        defined the size of train test split as 0.2
        splitted the file names as per the random train test split 
        Returns:
        - List of X_train and X_test
    """
    ```

## Contents

Contents of this folder are summarized as follows

| Item Name | Description |
| :---- | :---- |
| [set12_dataloader.py](./BSDS500.py) | Contains the main functions ` train_test_split_for_image_SET12` (back end) |
[![Developer: ParaB0Y(aman singh)](https://img.shields.io/badge/Developer-ParaB0Y(aman singh)-blue)](https://github.com/ParaB0Y)
