# Pipelines for Data Handling

Contains pipelines for handling data

## Table of contents

- [Pipelines for Data Handling](#pipelines-for-data-handling)
    - [Table of contents](#table-of-contents)
    - [Image Pipelines](#image-pipelines)
        - [Adding Noise](#adding-noise)
        - [Generate Patches](#generate-patches)
        - [Shuffling images](#shuffling-images)
    - [Contents](#contents)

## Image Pipelines

Pipelines for handling images

### Adding Noise

To add noise (artificial white gaussian noise) to images, use the following functions accordingly

1. `imgs_add_awgn`: Adds noise for a single distribution (mu and sigma) to the images and returns the result. The function header is

    ```py
    def add_gauss_noise_batch(imgs, sigma = 0.5, mu = 0.0, rngo = None):
    """
        Adds gaussian noise to a batch of images. Note that the same
        noise is added to each channel.

        Parameters:
        - imgs: list[np.ndarray]
            A list of images, each of shape (W, H, C), dtype: uint8
        - sigma: float      default: 0.5
            Standard deviation (spread or "width") of distribution
        - mu: float         default: 0.0
            Mean ("center" or location) of distribution
        - rngo: np.random.Generator     default: None
            The random number generator object to use (in case of
            consistency).
        
        Returns:
        - noisy_imgs: list[np.ndarray]
            A list of images, each of shape (W, H, C) with gaussian
            noise added. Same length as that of imgs. Range of each
            pixel is 0 to 255 ('uint8' dtype).
    """
    ```

2. `imgs_add_multinoise`: Adds noise for multiple levels of sigma (mu is same, = 0, for all). Returns the corresponding clean images as well (for correspondence sake). The function header is

    ```py
    def add_gauss_sigma_urange(imgs, minSigma = 0.1, maxSigma = 0.5,
    numSigmas = 5, mu=0.0, rngo = None):
    """
        Generates gaussian noises of different variance and applies
        them to the image. The sigma values are taken from a uniform
        distribution. Sigma is the standard deviation of the gaussian
        noise that'll be added

        Parameters:
        - imgs: list[np.ndarray]
            A list of images, each of shape (W, H, C)
        - minSigma: float       default: 0.1
            Minimum sigma value (can be 0)
        - maxSigma: float       default: 0.5
            Maximum sigma value
        - numSigmas: int
            Number of samples to take (for sigma). Each sample will
            correspond to one distribution.
        - mu: float         default: 0.0
            Mean ("center" or location) of all distributions
        - rngo: np.random.Generator     default: None
            The random number generator object to use (in case of
            consistency).
        
        Returns:
        - noisy_imgs: list[np.ndarray]
            A list of images, each of shape (W, H, C) with gaussian
            noise added. Length is 'length of imgs' * numSigmas.
        - clean_imgs: list[np.ndarray]
            The list 'imgs' repeated 'numSigma' times (for consistency
            sake). It is list multiplication.
    """
    ```

    In the current implementation, the function takes nearly a minute for generating 17 noise levels for about 400 images.

### Generate Patches

To generate patches (smaller sub-images) from a given image, use the following function accordingly

1. `impu23_extract`: Extract square patches from an image with aspect ratio 2:3 (landscape or portrait or mix). Requires the stride along major axis and minor axis. The function header is

    ```py
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
    ```

### Shuffling images

To shuffle images in a list, use the following function accordingly

1. `shuffle_corrimgs`: Shuffle images while maintaining index correspondences. The function header is

    ```py
    def shuffle_img_img_corres(imgs1: list[np.ndarray],
    imgs2 :list[np.ndarray], num_out = None):
    """
        Shuffles images in the lists (while maintaining corresponding
        indices).

        Parameters:
        - imgs1: list[np.ndarray]
            Contains `N` items, each being an image of shape (H, W, C)
        - imgs2: list[np.ndarray]
            Contains `N` items, each being an image of shape (H, W, C)
        - num_out: int or None      default: None
            The number of samples needed to output. Must be <= N.
            First these many indices are returned in both lists.
        
        Returns:
        - s_imgs1: list[np.ndarray]
            Contains N_out elements from imgs1 (shuffled), each being
            an image of the same shape as it was. `N_out` is `num_out`
            if it was passed, else it's `N`.
        - s_imgs2: list[np.ndarray]
            Contains N_out elements from imgs2 (shuffled), each being
            an image of the same shape as it was
    """
    ```

## Contents

Contains the following files

| Item Name | Description |
| :---- | :---- |
| [img_patches.py](./img_patches.py) | Image patches |
| [img_add_noise.py](./img_add_noise.py) | Noise addition to images |
| [shuffle_imgs.py](./shuffle_imgs.py) | Shuffling images |
