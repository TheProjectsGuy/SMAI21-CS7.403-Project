"""
    The BSDS500 Dataset Module

    Home page: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
"""

# %% Import modules
import os
import tarfile
import glob
import copy
from typing import TypedDict
import numpy as np
from PIL import Image
import scipy.io


# %% Dataset types
# Ground Truths
class GroundTruthType(TypedDict):
    # Number of annotations (per image)
    na: list[int]
    # Segmentation for every annotator (per image)
    segmentation: list[np.ndarray]
    # Boundaries for every annotator (per image)
    boundaries: list[np.ndarray]

# Dataset dictionary for training, testing or validation set
class SingleDataSetDict(TypedDict):
    images: list[np.ndarray]
    groundTruth: GroundTruthType

# Type of the entire dataset (as dictionary)
class DataSetDict(TypedDict):
    training: SingleDataSetDict
    test: SingleDataSetDict
    validation: SingleDataSetDict

# %% Main class
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
    __destination_folder = "./Data/"
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
    # Constructor
    def __init__(self, unzip = True) -> None:
        self.zip_location = os.path.abspath(os.path.expanduser(
            BSDS500_DataSet.__dataset_zip_file))
        self.zip_extract_loc = os.path.abspath(os.path.expanduser(
            BSDS500_DataSet.__destination_folder))
        self.data_uri = str(BSDS500_DataSet.__download_url)
        # Check if dataset exists
        if os.path.isfile(self.zip_location):
            print(f"Dataset BSDS500 found at '{self.zip_location}'")
            # If asked to unzip data
            if unzip:
                # # If path exists, clear it
                if os.path.exists(self.zip_extract_loc):
                    print("Path already exists, no extraction")
                else:
                    # Extract
                    os.mkdir(self.zip_extract_loc)
                    file = tarfile.open(self.zip_location)
                    file.extractall(self.zip_extract_loc)
                    file.close()
                    print(f"Extracted at: {self.zip_extract_loc}")
        else:
            # TODO: Maybe have `tf.keras.utils.get_file` do this
            raise FileNotFoundError(
                f"Dataset not present in '{self.zip_location}'. "
                f"Download it from '{self.data_uri}' at the location")
        # Ready to load data (manual call later)
        self.data : DataSetDict = dict()

    # Load the data (images, ground truth)
    def load_data(self) -> DataSetDict:
        """
            Loads all data from the data location and returns the
            dictionary.

            Returns:
            - data: dict
                A dictionary with keys in ["training", "validation",
                "test"]. Each key contains a dict with the keys
                - "images": list of images, each image is of type
                    np.ndarray and shape (H, W, C)
                - "groundTruth": A dictionary having the keys
                    - "na": Number of annotators for the samples (as a
                        list corresponding indices)
                    - "segmentation": list of list of annotated
                        segments for each sample
                    - "boundaries": List of list of annotated
                        boundaries for each sample
        """
        # Load image, segmentation, boundary sample
        def load_sample(img_path, gt_path):
            """
                Loads a single sample from the corresponding files

                Parameters:
                - img_path: Full path (w .jpg) for image
                - gt_path: Path for ground truth .mat file

                Returns:
                - x_img_np: np.ndarray  shape: (H, W, C)
                    Image having H rows, W columns and C channels
                - y_ann: int
                    Number of annotators for the sample (4 to 9)
                - y_seg: list[np.ndarray]
                    List of (H, W) images containing segments (as int)
                - y_bdr: list[np.ndarray]
                    List of (H, W) images for border (as int - 0, 1)
            """
            # Read image
            x_img_np = np.array(Image.open(img_path)) # (H, W, C)
            # Read .mat file for ground truth
            p_gt = scipy.io.loadmat(gt_path)["groundTruth"]
            y_ann: int = p_gt.shape[1]   # Number of annotators
            # Collect true labels (seg, bdr) from all annotators
            y_seg = [p_gt[0, i]["Segmentation"][0, 0] \
                for i in range(y_ann)]
            y_bdr = [p_gt[0, i]["Boundaries"][0, 0] \
                for i in range(y_ann)]
            return x_img_np, y_ann, y_seg, y_bdr

        # Path to data
        data_path = self.zip_extract_loc + "/" + \
            BSDS500_DataSet.__data_loc
        # --- Load training data ---
        x_imgs = []     # List of images
        y_anns = []     # Number of annotations
        y_segs = []     # List of list of segmentation annotations
        y_bdrs = []     # List of list of boundary annotations
        img_path = os.path.realpath(data_path + \
            BSDS500_DataSet.__x_tr_loc)
        gt_path = os.path.realpath(data_path + \
            BSDS500_DataSet.__y_tr_loc)
        img_samples: list[str] = glob.glob(img_path + "/*.jpg")
        gt_samples: list[str] = glob.glob(gt_path + "/*.mat")
        # Assuming files with the same number correspond
        img_samples.sort()
        gt_samples.sort()
        for img_s, gt_s in zip(img_samples, gt_samples):
            x, ya, ys, yb = load_sample(img_s, gt_s)
            # Record all data
            x_imgs.append(x)
            y_anns.append(ya)
            y_segs.append(ys)
            y_bdrs.append(yb)
        print(f"Loaded {len(y_anns)} training samples")
        # Save training data
        self.data["training"] = {
            "images": copy.deepcopy(x_imgs),
            "groundTruth": {
                "na": copy.deepcopy(y_anns),
                "segmentation": copy.deepcopy(y_segs),
                "boundaries": copy.deepcopy(y_bdrs)
            }
        }
        # --- Load test data ---
        x_imgs = []
        y_anns = []
        y_segs = []
        y_bdrs = []
        img_path = os.path.realpath(data_path + \
            BSDS500_DataSet.__x_ts_loc)
        gt_path = os.path.realpath(data_path + \
            BSDS500_DataSet.__y_ts_loc)
        img_samples: list[str] = glob.glob(img_path + "/*.jpg")
        gt_samples: list[str] = glob.glob(gt_path + "/*.mat")
        for img_s, gt_s in zip(img_samples, gt_samples):
            x, ya, ys, yb = load_sample(img_s, gt_s)
            # Record all data
            x_imgs.append(x)
            y_anns.append(ya)
            y_segs.append(ys)
            y_bdrs.append(yb)
        print(f"Loaded {len(y_anns)} test samples")
        # Save training data
        self.data["test"] = {
            "images": copy.deepcopy(x_imgs),
            "groundTruth": {
                "na": copy.deepcopy(y_anns),
                "segmentation": copy.deepcopy(y_segs),
                "boundaries": copy.deepcopy(y_bdrs)
            }
        }
        # --- Load validation data ---
        x_imgs = []
        y_anns = []
        y_segs = []
        y_bdrs = []
        img_path = os.path.realpath(data_path + \
            BSDS500_DataSet.__x_vl_loc)
        gt_path = os.path.realpath(data_path + \
            BSDS500_DataSet.__y_vl_loc)
        img_samples: list[str] = glob.glob(img_path + "/*.jpg")
        gt_samples: list[str] = glob.glob(gt_path + "/*.mat")
        for img_s, gt_s in zip(img_samples, gt_samples):
            x, ya, ys, yb = load_sample(img_s, gt_s)
            # Record all data
            x_imgs.append(x)
            y_anns.append(ya)
            y_segs.append(ys)
            y_bdrs.append(yb)
        print(f"Loaded {len(y_anns)} validation samples")
        # Save training data
        self.data["validation"] = {
            "images": copy.deepcopy(x_imgs),
            "groundTruth": {
                "na": copy.deepcopy(y_anns),
                "segmentation": copy.deepcopy(y_segs),
                "boundaries": copy.deepcopy(y_bdrs)
            }
        }
        return self.data

# %%
if __name__ == "__main__":
    raise ImportError("This script shouldn't be run as main")
