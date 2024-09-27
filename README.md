# Simple Description

This is a toolkit for evaluating medical imaging tasks.<br>
The evaluation metrics related to MedPy are in metrics/binary.py.<br>
Based on the MedPy package, we have made additions in metrics/metrics_all.py which include:<br>
accuracy, precision, recall, specificity, f1_score, dice_coefficient, iou, g_mean,
mae, hausdorff_distance, hausdorff_95, ssim, ncc, psnr, cohen_kappa, log_loss,
fpr, fnr, voe, rvd, sensitivity, jaccard_coefficient, tnr, tpr, confusion matrix, ROC, AUC, misclassification rate, MCC, FDR, NPV, balanced_accuracy,
mse, MI, NMI, CC, cross-entropy, FID
<br>
We retained the relevant parts of MedPy for processing input images in load.py.<br>

## Supported Image File Formats

Medical formats:<br>
ITK MetaImage (.mha/.raw, .mhd)<br>
NIfTI (.nia, .nii, .nii.gz, .hdr, .img, .img.gz)<br>
Analyze (.hdr/.img, .img.gz)<br>
Nearly Raw Raster Data (Nrrd) (.nrrd, .nhdr)<br>
Medical Imaging NetCDF (MINC) (.mnc, .MNC)<br>
Guys Image Processing Lab (GIPL) (.gipl, .gipl.gz)<br>
<br>Microscopy formats:<br>
Medical Research Council (MRC) (.mrc, .rec)<br>
Bio-Rad (.pic, .PIC)<br>
LSM (Zeiss) Microscopy Images  (.tif, .TIF, .tiff, .TIFF, .lsm, .LSM)<br>
Stimulate / Signal Data (SDT) (.sdt)<br>
<br>Visualization formats:<br>
VTK Images  (.vtk)<br>
<br>Other formats:<br>
Portable Network Graphics (PNG) (.png, .PNG)<br>
Joint Photographic Experts Group (JPEG) (.jpg, .JPG, .jpeg, .JPEG)<br>
Tagged Image File Format (TIFF) (.tif, .TIF, .tiff, .TIFF)<br>
Windows bitmap (.bmp, .BMP)<br>
Hierarchical Data Format (HDF5) (.h5 , .hdf5 , .he5)<br>
MSX-DOS Screen-x (.ge4, .ge5)<br>

## Return Types

Image data: Returned as a NumPy array, with dimensions ordered as x, y, z, c.<br>
<br>Header information: Returned as a Header object, containing metadata of the image.

## Error Handling

If the image file does not exist, an ImageLoadingError exception is thrown.


## Colab Demo

You can run the demo directly in Google Colab by clicking the link below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16DLfxqDoiTjslmxODYxyvW4yPDtbL4Ew?usp=sharing)

Or use this direct link: https://colab.research.google.com/drive/16DLfxqDoiTjslmxODYxyvW4yPDtbL4Ew?usp=sharing

## Installation
The code requires `python>=3.8`, as well as `pytorch` and `torchvision`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

You can use this package by 

```
pip install git+https://github.com/HanRu1/MedEvalKit.git
```
