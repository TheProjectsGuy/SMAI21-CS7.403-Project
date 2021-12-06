#N3Aggregation

This module is created to Computes neural nearest neighbors for image data based on extracting patches in strides.


In this folder, the `N3Aggregation2D` model is implemented. This is the  class for computing neural nearest neighbors

## Table of contents

- [N3Aggregation](#dncnn)
    - [Table of contents](#table-of-contents)
    - [Data](#data)
    -[Functions](#Functions)

## Data

The datasets used by the model above is [SET12] and [urban100]. The Model's goal is to  Computes neural nearest neighbors for image data based on extracting patches in strides.

For **Testing**:[SET12] and [urban100] , which is composed of 12 commonly used images and 100 images of urban scenes respectively for model evaluation and testing the model.Both dataset have test_split as 0.2.

For **Training**: The remaining 80 persent images are used for training.

### Functions

1. `compute_distances`:  Computes pairwise distances for all pairs of query items and potential neighbors.
    ```py
    def compute_distances():
    """
        param xe: BxNxE tensor of database item embeddings
        :param ye: BxMxE tensor of query item embeddings
        :param I: BxMxO index tensor that selects O potential neighbors for each item in ye
        :param train: whether to use tensor comprehensions for inference (forward only)
        
        Returns:
        - :return: a BxMxO tensor of distances
    """
    ```
2. `aggregate_output`:  Calculates weighted averages for k nearest neighbor volumes.
    ```py
    def compute_distances():
    """
        Calculates weighted averages for k nearest neighbor volumes.
        :param W: BxMxOxK matrix of weights
        :param x: BxNxF tensor of database items
        :param I: BxMxO index tensor that selects O potential neighbors for each item in ye
        :param train: whether to use tensor comprehensions for inference (forward only)
        :return: a BxMxFxK tensor of the k nearest neighbor volumes for each query item
    """
    ```

3. `CLASS N3AggregationBase`:   Domain agnostic base class for computing neural nearest neighbors.
    ```py
    def init():
    """
        :param k: Number of neighbor volumes to compute
        :param temp_opt: options for handling temperatures, see `NeuralNearestNeighbors`
    """
    def forward
    """
        :param x: database items, shape BxNxF
        :param xe: embedding of database items, shape BxNxE
        :param ye: embedding of query items, shape BxMxE
        :param y: query items, if None then y=x is assumed, shape BxMxF
        :param I: Indexing tensor defining O potential neighbors for each query item
            shape BxMxO
        :param log_temp: optional log temperature
        :return
        it->
          # compute distance
          # compute aggregation weights
    """
    ```
4. `CLASS N3Aggregation2D`:    Computes neural nearest neighbors for image data based on extracting patches
    in strides.
    ```py
    def __init():
    """
        :param indexing: function for creating index tensor
        :param k: number of neighbor volumes
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param temp_opt: options for handling temperatures, see `NeuralNearestNeighbors`
    """
    def forward()
    """
        :param x: database image
        :param xe: embedding of database image
        :param ye: embedding of query image
        :param y: query image, if None then y=x is assumed
        :param log_temp: optional log temperature image
        :return:
        it->
         # Convert everything to patches
         # Get nearest neighbor volumes
        # Convert patches back to whole images  
    """ 
    ```


![Developer: ParaB0Y(aman singh)](https://img.shields.io/badge/Developer-ParaB0Y(aman singh)-blue)
