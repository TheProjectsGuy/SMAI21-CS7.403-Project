The flow of execution is as follows :
1)BSDS500_loader_and_patch_generator.ipynb : This notebook loads the BSDS500 dataset and generates patches for the train and test sets
2)Mixing_of_noise_in_dataset : This notebook is used to mix the noise to the patches created in step -1 and then recreate the training and testing set for the FFDnet
3)The FFDnet architecture : This notebook contains the architecture for FFDnet as suggested in the paper for both training the color as well as the grayscale image dataset

Note: Currently the patch size selected is 40 x 40
However for training the FFDnet on colored images it needs to be changed to 50 x 50 ( since teh same is mentioned in paper amd I want to replicate the experiment exactly)
Hence in this case input vector size needed is  50x50x3 whereas we currenly have 40 x 40 x3
This needs to be changed in the previous code files similarly for grayscale image the Patch size will have to become 70 x 70 and hence the input size will change likewise

These changes shall be made before training the model

4)Recreate_Image_from_patches : This Notebook is used to recerate the image from the patches after the noise is removed from patches.


Upcoming work : After the training is done we need to compare the orignal image with the recerated image.