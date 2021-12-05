# Functions to handle image patches
"""
    Functions that handle patch extraction from images

    Creator: TheProjectsGuy
"""

# %% Import everything
import numpy as np

# %% Function definitions
# Extract patches from images with varying dimensions
def batch_u23lp_sextract(imgs: list[np.ndarray], ps: int,
    stride = [1, 1]):
    """
        Batch - Uniform, 2:3 aratio, [L]andscape | [P]ortrait - square
        patch extraction

        Extract square patches from a list of images. Each image is of
        2:3 aspect ratio and is either taken in landscape or in
        portrait mode. The patch extraction is uniform across all
        images (patches are uniformly sampled). Each patch is a of
        square shape. The strides are computed internally. If the
        aspect ratio is not exact, the algorithm makes adjustments for
        the aspect ratio to become exact (by removing boundary).

        Parameters:
        - imgs: list[np.ndarray]
            A list of images. Each element must be of shape (H, W, C)
            where H is height, W is width and C is number of channels
            in the image.
        - ps: int
            The patch size. Patches cropped from each image are of
            square size (ps, ps).
        - stride: list[int]
            The strides for [major, minor] dimension for extracting
            patches. It is single step by default.
            **WARN**: Currently, if a dimension goes out of range when
                extracting patches, an error is thrown (out of index)
        
        Returns:
        - aps: list[np.ndarray]
            All patches as a list. Each element has shape (ps, ps, C).
            The number of patches are determined by the strides.
    """
    # Parameters
    H, W, c = imgs[0].shape # Image dimensions
    p = int(ps) # Patch size (p, p)
    sM, sm = stride # [Major, Minor] stride lengths
    aps = []    # All patches foun
    for img in imgs:    # For each image
        # Check shape (Height, Width, #-channels)
        H, W, c = img.shape
        if H > W:   # Portrait image (more strides along height)
            sh = sM
            sw = sm
        else:   # Landscape image (more strides / samples along width)
            sh = sm
            sw = sM
        # Take samples
        imps = []   # Patches from this image
        for h in range(0, H-p, sh):   # Along height
            for w in range(0, W-p, sw):   # Along width
                # Add patch
                imps.append(img[h:h+p, w:w+p, :])
        # Add to total
        aps += imps
    return aps

# %%
