# Whole Slide Cervical Cancer Screening Using Graph Attention Network and Supervised Contrastive Learning

This repository contains the implementation of the framework and experiments described in "Whole Slide Cervical Cancer Screening Using Graph Attention Network and Supervised Contrastive Learning", which is submitted to MICCAI 2022.

Authors: Xin Zhang, Maosong Cao, Sheng Wang, Jiayin Sun, Xiangshan Fan, Qian Wang, Lichi Zhang



## Requirements

To download all prerequisites, in the terminal type
`pip install -r requirements.txt`



## Example usage

`python train.py` will train the GAT model with contrastive learning.

 The results of training curves, classification performance, t-SNE and the models will be generated in folder `output`, which is automatically created after training. 

 `python test.py` can be used to test the trained models on both our sample data and your own data.

Before running `python train.py`, you may need to detect the suspicious cells and use an encoder to extract the features of patches. The source codes for detection and classification (including the encoder) are released in `Suspicious_cell_detection` and `Patch_classification`.

## Experimental results of different patch numbers
 

<table>
    <tr>
        <td>patch number</td>
        <td>ACC</td>
        <td>AUC</td>
        <td>REC</td>
        <td>PREC</td>
        <td>F1</td>
    </tr>
    <tr>
        <td>10</td>
        <td>84.76±1.02</td>
        <td>91.86±0.93</td>
        <td>82.17±1.37</td>
        <td>86.17±1.40</td>
        <td>84.40±0.95</td>
    </tr>
    <tr>
        <td>20</td>
        <td>85.79±1.21</td>
        <td>92.52±0.91</td>
        <td>82.63±2.04</td>
        <td>88.15±1.39</td>
        <td>85.28±1.27</td>
    </tr>
    <tr>
        <td>30</td>
        <td>85.61±0.96</td>
        <td>92.34±1.04</td>
        <td>82.75±2.07</td>
        <td>87.72±1.39</td>
        <td>85.13±1.02</td>
    </tr>
</table>

All the results are conducted in 5-fold cross-validation as our paper illustrated.

## Qualitative results

The qualitative result analysis for the proposed method is presented in `qualitative results.pdf`.

## Dataset

Part of the data is shown in `data/Sample of Image Tiles`, `data/Sample of patches` and `data/Sample of graphs`. 

