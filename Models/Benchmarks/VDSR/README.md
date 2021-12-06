Single image super-resolution (SISR): The problem of generating a high-resolution image given a low-resolution using VDSR network. 


The process of training is as follows :

     Data loading from the 'train.h5' file generated using the dataloder 'BSD500_VDSR'.

     It is done using the function defined in the class 'DatasetFromHdf5' in SISR.ipynb.
     
     It loads the input data (patches generated from the images) and the target data (Ground truth patches )
     
     The patch size selected here is 40 x 40, however it needs to be changed to 80 x 80(mentioned in the paper which we want to replicate)
     
     From each image we extract 3833 patches of size 40 x 40.
     
     This data is fed to the VDSR network defined in the class 'Net' in the SISR.ipynb
     
     


