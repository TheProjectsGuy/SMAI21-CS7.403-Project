Model: VDSR

Task: Single Image Super Resolution

Dataloader: To get the data ready for training, validation, and testing with preprocessing and extracting the patches done.

     Datasets are:
     
       1. 'Berkeley  Segmentation Dataset 500 (BSD500)' [http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz],
            There are 500 RGB images in the dataset in JPG format (200 for training + 200 for testing + 100 for validating).
            Dimension of each image -  (321 x 481) or (481 x 321) 
     
       2. 'Urban100' [https://huggingface.co/datasets/eugenesiow/Urban100],
       
       3. 'SET5' [https://huggingface.co/datasets/eugenesiow/Set5]
     
First download the datasets from the links provided above. Extract the zip file and get the folder path.

Use the code 'data_visualize.py' to load the above datasets and visualize them by plotting some images.

As a part of preprocessing, from eah image, crop 3833 patches of size 40×40 and apply data augmentation by flipping and using a rotation∈{0◦,90◦,180◦,270◦} and using different scales of resolution [2,3,4].

This can be done using the code 'generate_train.m', and generate_test_mat.m which are matlab codes.

Both the codes 'generate_train.m' and generate_test_mat.m use the user defined matlab functions 'modcrop.m' and 'store2hdf5.m' that are provided in the same folder.
As a result, we get 'train.h5' and 'test.h5' files containing the image patches. 
