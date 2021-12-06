#dev=nayan
The flow of execution is as follows :
1)BSDS500_loader_and_patch_generator.ipynb : This notebook loads the BSDS500 dataset and generates patches for the train and test sets
2)Mixing_of_noise_in_dataset : This notebook is used to mix the noise to the patches created in step -1 and then recreate the training and testing set for the FFDnet
3)The FFDnet architecture : This notebook contains the architecture for FFDnet as suggested in the paper for both training the color as well as the grayscale image dataset
4)Recreate_Image_from_patches : This Notebook is used to recerate the image from the patches after the noise is removed from patches.

