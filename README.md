# Environment Setup

Python 3.9

Torch 2.1.0

CUDA 12.1

# Usage

Train DEN-Unet and validation:

`python train.py`

Test:

`python test.py`

# Data

## LIDC-IDRI

The LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative) dataset is a publicly available CT scan dataset designed for lung nodule detection, segmentation, and classification tasks. It contains thoracic CT scans of 1,018 patients, with each scan annotated by up to four experienced radiologists.

The dataset can be accessed and downloaded here: https://www.cancerimagingarchive.net/collection/lidc-idri/

    -input
      -dsb128
        -images
          -
        -masks

## Data Preparation

### Lung Parenchyma Extraction and Format Conversion

The raw image data in the LIDC-IDRI dataset is stored in DICOM format, which is not directly suitable for deep learning and contains substantial non-lung tissue that may hinder lesion learning. Therefore, the images are converted to npy format and processed with a lung segmentation algorithm to extract the lung parenchyma, improving the model's focus and performance.

This preprocessing step can be referred to from the following public resources and tools: https://github.com/jaeho3690/LIDC-IDRI-Preprocessing

After extracting and saving the processed lung images in .npy format, the next step is to convert the .npy files to .png images to match the input format requirements of most deep learning models. https://github.com/Sen182/Preprocess/blob/main/npy2img.py

### Image Cropping

Most nodules in the LIDC-IDRI dataset are small, occupying only a minor portion of each slice. Directly using full slices for training may cause the model to focus on irrelevant background, affecting classification and segmentation accuracy. To address this, the nodule masks provided by the dataset are used to crop the original images, extracting local regions containing lesions and resizing them to 128 × 128 pixels. This enables the model to concentrate on the nodule region, reduces background interference, improves nodule identification, and ensures consistent input data, supporting stable training and lower computational cost. https://github.com/Sen182/Preprocess/blob/main/get128.py
