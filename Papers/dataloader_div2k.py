import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    Images = np.array(images)
    return Images
