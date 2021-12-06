# Functions to handle shuffling or iterables
"""
    Functions to shuffle images (while keeping correspondence)

    Creator: TheProjectsGuy
"""

# %% Import everything
import numpy as np

# %% Function definitions
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
    # Number of images
    N = len(imgs1)
    N_out = int(N if num_out is None else num_out)
    # Index values (shuffle N)
    ind_vals = np.arange(N)
    np.random.shuffle(ind_vals)
    # Get shuffled image lists
    s_imgs1 = [imgs1[i] for i in ind_vals]
    s_imgs2 = [imgs2[i] for i in ind_vals]
    # Return the first N_out images
    return s_imgs1[:N_out], s_imgs2[:N_out]

# %%
