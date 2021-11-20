# Berkeley Segmentation Dataset 500

BSDS500 (Berkley Segmentation Dataset) is a dataset primarily for Image segmentation and boundary detection.

**Home Page**: From [Berkley Computer Vision Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html), you can download the entire set from [here][dataset-link]. You can find it on [paperswithcode](https://paperswithcode.com/dataset/bsds500).

## Table of contents

- [Berkeley Segmentation Dataset 500](#berkeley-segmentation-dataset-500)
    - [Table of contents](#table-of-contents)
    - [About the dataset](#about-the-dataset)
    - [Description](#description)
    - [Download](#download)

## About the dataset

BSDS500 was made for aiding research on image segmentation and boundary detection algorithms. It contains images and ground truth segments and boundaries. The description about each of them is given below

1. **Images**: There are 500 images in the dataset (200 for training + 200 for testing + 100 for validating). Each image is either `321, 481` or `481, 321` (represented as `height, width`) and in channel RGB (8-bit color). These are stored as JPG files.
2. **Ground Truth**: These are the segments as well as boundaries for all the images. The annotation task was carried by multiple annotators, and data from each of them is included. Each annotator gives their own _Segmentation_ estimate and _Boundaries_ estimate (the main target / task for which this dataset was created). These are stored (sample-wise) in `*.mat` files (MATLAB 5.0, Platform PCWIN). We're interested in the `groundTruth` variable in this file.
    1. Each sample includes a list of annotators (4 to 9, but 5 on an average; the shape is [1, `num_annotators`] of each sample).
    2. Each annotator (accessed using `[0, i]`) has a [structured datatype object](https://numpy.org/doc/stable/user/basics.rec.html) of type `dtype([('Segmentation', 'O'), ('Boundaries', 'O')])`. This is indexed using the string name (`['Segmentation']` or `['Boundaries']`) and has only one object each (follow with `[0, 0]`).
    3. Each object, whether under segmentation or boundary, has the same shape as the corresponding image.
        1. For a `'Segmentation'` object, the pixel values indicate the _segment_ (group number integer) it belongs to. This number starts from 1 and can go up to as many segments the annotator has detected.
        2. For a `'Boundaries'` object, the pixel values are either 0 (not a boundary pixel) or 1 (a boundary pixel).

    Example 1: Say we have loaded a `*.mat` file (containing annotations of an image) and retrieved the `groundTruth` variable in `y_gt`. We can retrieve the segmentation annotations of the third annotator by doing: (note that indices start from 0)

        ```py
        y_gt[0, 2]["Segmentation"][0, 0]
        ```

    Example 2: We can retrieve the boundaries annotations of the second annotator by doing

        ```py
        y_gt[0,1]["Boundaries"][0, 0]
        ```

## Description

## Download

You can download the dataset from [this link][dataset-link], or you could run the specified command

For Windows **PowerShell**

```pwsh
Invoke-WebRequest -Uri http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -OutFile BSR_bsds500.tgz
```

[dataset-link]: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
