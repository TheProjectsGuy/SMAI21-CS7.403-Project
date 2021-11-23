# Functions to handle (artificial) image noising
"""
    Functions that add noise to images
"""

# %% Import everything
import numpy as np

# %% Function definitions
# Add white gaussian noise to a set of images
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
    noisy_imgs = [] # Store noisy images
    # Default random number generator (new method)
    rng: np.random.Generator = np.random.default_rng() \
        if rngo is None else rngo
    # Add noise to each image
    for im in imgs:
        img = im.copy()
        nlayer = rng.normal(mu, sigma, (img.shape[0], img.shape[1]))
        # Add noise to image
        nimg = img + np.stack([nlayer]*3, axis=2)
        # Normalize it
        n_img = (255*(nimg - np.min(nimg))/np.ptp(nimg)).astype(
            np.uint8)
        noisy_imgs.append(n_img)
    # Return noisy images
    return noisy_imgs

# Add gaussian noise with uniform (different) sigma values in a range
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
    noisy_imgs = [] # Store noisy images
    # Default random number generator (new method)
    rng: np.random.Generator = np.random.default_rng() \
        if rngo is None else rngo
    sigma_vals = rng.uniform(minSigma, maxSigma, numSigmas)
    # Add AWGN images
    for sig in sigma_vals:
        noisy_imgs += add_gauss_noise_batch(imgs, sig, mu, rng)
    return noisy_imgs, imgs * numSigmas


# %%
