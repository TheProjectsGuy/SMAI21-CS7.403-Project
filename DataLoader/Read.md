This project uses two different datasets. To make the task of data loading easier, this folder contains scripts that contain data loading functions.  For training , we took 200 images of the BSD500 training set and cropped 3833 patches of size 40×40 from each image and apply data augmentation by flipping and using a rotation∈{0◦,90◦,180◦,270◦}.
In this folder we have the main code 'data_visualize.py' to read the BSD500 dataset. 
This folder also have the codes 'generate_train.m', and 'generate_test_mat.m' to generate the augmented data for training of vdsr.
