# Flexidot-Classifier

## Introduction
Flexidot-Classifier provides code to the publication [placeholder].
It serves as a tool for automatically classifying the output of [FlexiDot](https://github.com/flexidot-bio/flexidot) (or other dotplots) using algorithmic methods of image analysis and computer vision.
Specifically, it reads PDF or PNG files, extracts single plots, and analyzes the image information of the dotplots.
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

1. Clone the repository (git clone https://github.com/tudipffmgt/Flexidot-Classifier.git)
2. Create a virtual environment and install Python 3.12 or newer
3. The dependencies are provided in the `requirements.txt` (pip install -r requirements.txt)

### Step 1: Prepare Input Files

Create an `input/` folder in the project directory and add your Flexidot plot files.

**Supported input modes:**

| Mode | Description |
|------|-------------|
| Single multi-page PDF | One PDF file containing multiple plot pages |
| Multiple PDFs | Several single-page PDF files |
| Multiple PNGs | Several PNG images (one plot page per image) |

**Note:** Do not mix PDFs and PNGs in the same folder.

### Step 2: Run the Classifier

`python main.py`

### Step 3: View Results

Results are saved in `output/final/`:

| File | Description |
|------|-------------|
| `*_annotated.png` | Input images with detected elements highlighted |
| `ltr.txt` | List of detected LTR retrotransposons (sequence numbers) |
| `tandem.txt` | List of detected tandem repeats (sequence numbers) |
| `satellite.txt` | List of detected satellites (sequence numbers) |

## Detection Colors

| Element | Color |
|---------|-------|
| LTR Retrotransposon | Teal |
| Tandem Repeat | Cyan |
| Satellite | Orange |

## Configuration

Detection parameters can be adjusted in the `Config` class in `main.py`:

    @dataclass
    class Config:
        # Crop margins (pixels, calibrated for 750x750 masks)
        crop_bottom_left: int = 37
        crop_top_right: int = 15
        
        # Element detection thresholds
        edge_threshold: float = 0.05
        min_contours_ltr: int = 3
        min_contours_tandem: int = 5
        min_contours_satellite: int = 7

