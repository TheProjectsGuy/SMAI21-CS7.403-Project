# Functions to load the BSDS500 dataset
"""
    Contains functions to load the BSDS500 dataset
"""

# %% Import everything
import os

try:
    # For local use
    from BSDS500 import BSDS500_DataSet
    from BSDS500 import DataSetDict
except ModuleNotFoundError:
    # For module use
    from DataLoaders.BSDS500 import BSDS500_DataSet
    from DataLoaders.BSDS500.BSDS500 import DataSetDict


# %% Main functions
# Load BSDS500
def load_bsds500_dataset():
    """
        Calls the load_data function of the dataset and returns the
        dictionary (as it is)

        Returns:
        - data_dict: DataSetDict    The dataset dictionary
    """
    return BSDS500_DataSet().load_data()

# Load images (all of them)
def load_bsds500_images(ds_obj = None):
    """
        Loads all 500 images of the BSDS 500 dataset (as single batch)

        Paraemters:
        - ds_obj: DataSetDict       default: None
            The loaded data as dictionary. If None, the function loads
            fresh data.

        Returns:
        - imgs: list[np.ndarray]
            List of 500 images, each of shape (H, W, C = 3)
    """
    # Load dataset
    _ds: DataSetDict = load_bsds500_dataset() if ds_obj is None \
        else ds_obj
    imgs = []   # Stores all images
    imgs += _ds["training"]["images"]
    imgs += _ds["test"]["images"]
    imgs += _ds["validation"]["images"]
    # Return them
    return imgs

# BSD68 file names
def val_bsd68_img_names():
    """
        Returns a list of validation images (for the BSD68 set)

        Returns:
        - img_fnames: list[str]
            All file names (*.jpg) for images used in BSD68
    """
    __val68_file = "BSDS_val68_list.txt"
    _dir = os.path.dirname(__file__)
    bsd68_inames_file = f"{_dir}/{__val68_file}"
    img_fnames = []
    # Read file and append to list
    fhdlr = open(bsd68_inames_file, 'r')
    img_names = fhdlr.readlines()
    fhdlr.close()
    # Remove the trailing '\n'
    img_fnames = list(map(lambda x: x[:-1], img_names))
    return img_fnames

# Load the training and test split (BSD68 test)
def load_imgs_432_68_split(ds_obj = None):
    """
        Loads the BSDS500 images with the recommended split of 432
        images for training and 68 images for testing. The 68 images
        are actually from the validation split of the original BSDS500

        Paraemters:
        - ds_obj: DataSetDict       default: None
            The loaded data as dictionary. If None, the function loads
            fresh data.

        Returns:
        - train_imgs: list[np.ndarray]  Images for training (len=432)
        - test_imgs: list[np.ndarray]   Images for testing (len=68)
    """
    # Dataset object (load fresh if not existing)
    _ds: DataSetDict = load_bsds500_dataset() if ds_obj is None \
        else ds_obj
    # Get the images
    imgs_train = _ds["training"]["images"]
    imgs_test = _ds["test"]["images"]
    imgs_val = _ds["validation"]["images"]
    # Train and test are straightaway for 'training'
    train_imgs = imgs_train + imgs_test   # Add lists
    # Search for BSD68 indices in validation split of BSD500
    bsd68_imgs_fnames = val_bsd68_img_names()  # File names
    val_paths = _ds["validation"]["paths"]["img"]   # 100 images
    val_items = list(map(os.path.basename, val_paths))
    finds = [val_items.index(x) for x in bsd68_imgs_fnames] # Indices
    test_imgs = [] # BSD68 images
    val_train_imgs = [] # Images from validation used for 'training'
    # Filter validation images
    for i, img in enumerate(imgs_val):
        if i in finds:  # BSD68 sample
            test_imgs.append(img)
        else:
            val_train_imgs.append(img)
    # Add trainable validation images
    train_imgs += val_train_imgs
    return train_imgs, test_imgs

# Load the images data
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
    if split_data:
        return load_imgs_432_68_split(ds_obj)
    else:
        return load_bsds500_images(ds_obj)

# %% Main
if __name__ == "__main__":
    raise ImportError("This script shouldn't be run as main")

# %% Experiments

# %%
