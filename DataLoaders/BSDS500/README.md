# Berkeley Segmentation Dataset 500

BSDS500 (Berkley Segmentation Dataset) is a dataset primarily for Image segmentation and boundary detection.

**Home Page**: From [Berkley Computer Vision Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), you can download the entire set from [here][dataset-link]. You can find it on [paperswithcode](https://paperswithcode.com/dataset/bsds500).

## Table of contents

- [Berkeley Segmentation Dataset 500](#berkeley-segmentation-dataset-500)
    - [Table of contents](#table-of-contents)
    - [About the dataset](#about-the-dataset)
        - [BSD68 Dataset](#bsd68-dataset)
    - [Download](#download)
    - [Module Usage](#module-usage)
        - [Functions](#functions)
        - [Class](#class)
        - [Typing](#typing)
    - [Contents](#contents)

## About the dataset

BSDS500 was made for aiding research on image segmentation and boundary detection algorithms. It contains images and ground truth segments and boundaries. The description about each of them is given below

1. **Images**: There are 500 images in the dataset (200 for training + 200 for testing + 100 for validating). Each image is either `321, 481` or `481, 321` (represented as `height, width`) and in channel RGB (8-bit color). These are stored as JPG files.
2. **Ground Truth**: These are the segments as well as boundaries for all the images. The annotation task was carried by multiple annotators, and data from each of them is included. Each annotator gives their own _Segmentation_ estimate and _Boundaries_ estimate (the main target / task for which this dataset was created). These are stored (sample-wise) in `*.mat` files (MATLAB 5.0, Platform PCWIN). We're interested in the `groundTruth` variable in this file.
    1. Each sample includes a list of annotators (4 to 9, but 5 on an average; the shape is [1, `num_annotators`] of each sample).
    2. Each annotator (accessed using `[0, i]`) has a [structured datatype object](https://numpy.org/doc/stable/user/basics.rec.html) of type `dtype([('Segmentation', 'O'), ('Boundaries', 'O')])`. This is indexed using the string name (`['Segmentation']` or `['Boundaries']`) and has only one object each (follow with `[0, 0]`).
    3. Each object, whether under segmentation or boundary, has the same shape as the corresponding image.
        1. For a `'Segmentation'` object, the pixel values indicate the _segment_ (group number integer) it belongs to. This number starts from 1 and can go up to as many segments the annotator has detected.
        2. For a `'Boundaries'` object, the pixel values are either 0 (not a boundary pixel) or 1 (a boundary pixel).

    Example 1: Say we have loaded a `*.mat` file (containing annotations of an image) and retrieved the `groundTruth` variable in `y_gt`. We can retrieve the segmentation annotations of the third annotator by doing: (note that indices start from 0)

        ```py
        y_gt[0, 2]["Segmentation"][0, 0]
        ```

    Example 2: We can retrieve the boundaries annotations of the second annotator by doing

        ```py
        y_gt[0,1]["Boundaries"][0, 0]
        ```

### BSD68 Dataset

There are 68 images taken from the validation dataset of the original BSDS500 dataset. The file names are listed in [BSDS_val68_list.txt](./BSDS_val68_list.txt). This file is directly borrowed from [visinf/n3net](https://github.com/visinf/n3net).

## Download

You can download the dataset from [this link][dataset-link], or you could run the specified command

For Windows **PowerShell**

    ```pwsh
    Invoke-WebRequest -Uri http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -OutFile BSR_bsds500.tgz
    ```

[dataset-link]: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz

## Module Usage

The module has functions (and a class) to dedal with the BSDS500 dataset. The functions are generally sufficient. 

> **Note**: This module, by default, assumes that the ZIP file is located at `~/Downloads/Datasets/BSR_bsds500.tgz` (class setting). If a folder is existing under the name `~/Downloads/Datasets/BSDS500/`, the extraction will not happen.

### Functions

The module has the following functions that can be used accordingly to load the dataset

1. `load_data`: Loads the entire BSDS500 dataset (images, ground truths, paths, everything related to the dataset). The header of this function is described below

    ```py
    def load_bsds500_dataset():
    """
        Calls the load_data function of the dataset and returns the
        dictionary (as it is)

        Returns:
        - data_dict: DataSetDict    The dataset dictionary
    """
    ```

2. `load_imgs_dataset`: Loads only the images part (ground truth is discarded). Provision is there to retrieve the BSD68 (which is 68 images from validation) as test and the remaining 432 as training images. The function header is as follows

    ```py
    def load_imgs_dataset(ds_obj = None, split_data = True):
    """
        Loads the BSDS500 dataset (images only). Training and testing
        split is decided based on the BSD68 dataset.

        Paraemters:
        - ds_obj: DataSetDict       default: None
            The loaded data as dictionary. If None, the function loads
            fresh data. This is usually through BSDS500().load_data()
        - split_data: bool      default: True
            If true, a pair (train_imgs and, test_imgs) is returned,
            else the entire 500 images are returned as a list. The
            splitting happens based on the BSD68 dataset (432 training
            and 68 testing images). If there is no split, then all
            500 images are returned.
        
        Returns:
        IF split_data == True
            - train_imgs: list[np.ndarray]  Images for training
            - test_imgs: list[np.ndarray]   Images for testing
        ELSE
            - imgs: list[np.ndarray]    All 500 images of BSDS500
    """
    ```

### Class

The module has a class which handles the dataset finding, extraction and loading. However, it is meant for complete BSDS500 (and nothing more). This class is exposed as `BSDS500_DataSet` and will probably be used very little. The header of this class has important variables (do not change unless you know what you're doing), as shown below

    ```py
    class BSDS500_DataSet:
        """
            Main class for the dataset handling. Initializing object
            ensures that the data is present, call the `load_data`
            function to load (and return) data.

            Constructor Arguments:
            - unzip: bool
                If True, the program also ensures that the *.tgz file is
                extracted.
            
            Throws:
            - FileNotFoundError if dataset not available in path
        """
        # ZIP file location (desired) on system
        __dataset_zip_file = "~/Downloads/Datasets/BSR_bsds500.tgz"
        # Destination folder (for extracting)
        __destination_folder = "~/Downloads/Datasets/BSDS500/"
        # URL for dataset
        __download_url = r"http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
        # Data location (in extract)
        __data_loc = "BSR/BSDS500/data/" # Data folder
        # Input data in location (relative to __data_loc)
        __x_tr_loc = "images/train/"
        __x_ts_loc = "images/test/"
        __x_vl_loc = "images/val/"
        # Output data in location (relative to __data_loc)
        __y_tr_loc = "groundTruth/train/"
        __y_ts_loc = "groundTruth/test/"
        __y_vl_loc = "groundTruth/val/"
    ```

### Typing

The following types have been declared in the module ([python typing](https://docs.python.org/3/library/typing.html) for IDEs and inference)

1. `DataSetDict`: The dataset is returned through `load_data` in this dictionary format. It contains all data split into training (200 samples), testing (200 samples) and validation (100 samples).

    ```py
    class DataSetDict(TypedDict):
        training: SingleDataSetDict
        test: SingleDataSetDict
        validation: SingleDataSetDict
    ```

2. `SingleDataSetDict`: A particular segment / split is described as follows

    ```py
    class SingleDataSetDict(TypedDict):
        images: list[np.ndarray]
        groundTruth: GroundTruthType
        paths: PathsDict
    ```

3. `GroundTruthType`: The ground truth information is retrieved and stored in the following manner

    ```py
    class GroundTruthType(TypedDict):
        # Number of annotations (per image)
        na: list[int]
        # Segmentation for every annotator (per image)
        segmentation: list[np.ndarray]
        # Boundaries for every annotator (per image)
        boundaries: list[np.ndarray]
    ```

4. `PathsDict`: The full file paths for each image (`*.jpg`) and ground truth (`*.mat`) is also stored (just in case needed).

    ```py
    class PathsDict(TypedDict):
        img: list[str]
        gt: list[str]
    ```

## Contents

Contents of this folder are summarized as follows

| Item Name | Description |
| :---- | :---- |
| [BSDS500.py](./BSDS500.py) | Contains the main class `BSDS500_DataSet` (back end) |
| [def_load_dataset.py](./def_load_dataset.py) | Contains functions to load the dataset (front end) |
| [BSDS_val68_list.txt](./BSDS_val68_list.txt) | A file to describe the file-names of the BSD68 validation images |
