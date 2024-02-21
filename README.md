# Flexidot-Classifier

## Introduction
Flexidot-Classifier provides code to the publication [placeholder].
It serves as a tool for automatically classifying the PDF-output of [flexidot](https://github.com/molbio-dresden/flexidot) (or other dotplots) using algorithmic methods of image analysis and computer vision.
Specifically, it reads the PDF files, converts the plots to single PNG files, and extracts the image information of the dotplots.
At the moment it automatically classifies LTRs, satellites and tandems.
However, the parameters can be adapted to extract any kind of patterns.

If you are using this project for your research, please cite:
```
@article{123,
    author = {author},
    title = {title}
```

## Installation and dependencies

Steps for installation are the following:

1. Clone the repository
2. Create a virtual environment and install Python 3.12
3. The dependencies are provided in the `requirements.txt`.
4. For Windows users it is required to install poppler and change the path environment variable