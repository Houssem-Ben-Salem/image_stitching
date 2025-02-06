# Image Stitching Methods Comparison

This project implements and compares different image stitching methods, including traditional approaches (SIFT, ORB, AKAZE) and deep learning-based methods (SuperPoint + SuperGlue). It provides both a comprehensive evaluation framework and an interactive web interface for testing different methods.

## Authors
- Ben Salem Houssem
- Sauvé Catherine

## Features

- Multiple stitching methods:
  - SIFT (Scale-Invariant Feature Transform)
  - ORB (Oriented FAST and Rotated BRIEF)
  - AKAZE (Accelerated-KAZE)
  - SuperPoint + SuperGlue (Deep Learning-based)
- Comprehensive evaluation metrics:
  - Mean Corner Error (MCE)
  - Reprojection Error
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)
  - Processing time and memory usage
- Interactive web interface using Gradio
- Batch processing capabilities
- Detailed visualization of results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Houssem-Ben-Salem/image_stitching.git
cd image_stitching
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download SuperPoint + SuperGlue weights:
```bash
# Create directory for weights
mkdir -p SuperGluePretrainedNetwork/models/weights/

# Download weights (you'll need to get these from the official repository)
# Place them in SuperGluePretrainedNetwork/models/weights/
# - superglue_indoor.pth
# - superglue_outdoor.pth
```

## Project Structure

```
image-stitching/
├── methods/
│   ├── traditional/
│   │   ├── sift.py
│   │   ├── orb.py
│   │   └── akaze.py
│   ├── deep_learning/
│   │   └── superpoint.py
│   └── base.py
├── utils/
│   ├── data_loader.py
│   └── evaluation.py
├── experiments/
│   ├── evaluate_methods.py
│   └── compare_methods.py
├── app.py
├── requirements.txt
└── README.md
```

## Usage

### Running the Web Interface

```bash
python app.py
```
Then open your browser and navigate to `http://localhost:7860`

### Running Evaluations

To evaluate all methods on the full dataset:
```bash
python experiments/evaluate_methods.py
```

To evaluate on a subset of data:
```bash
python experiments/evaluate_methods.py --max-percentage 50
```

### Visualizing Comparisons

To visualize the comparison between methods:
```bash
python experiments/compare_methods.py
```

# 🔧 Configuration

Each method can be configured through its respective config file:
```python
sift_config = {
    'nfeatures': 0,
    'n_octave_layers': 3,
    'contrast_threshold': 0.04,
    # ...
}
```

## Evaluation Metrics

- **Mean Corner Error (MCE)**: Measures the average distance between transformed corners using estimated vs. ground truth homography
- **Reprojection Error**: Average distance between matched points after transformation
- **SSIM**: Measures structural similarity between stitched regions
- **PSNR**: Measures peak signal-to-noise ratio in overlapping regions
- **Time**: Processing time for feature detection, matching, and stitching
- **Memory**: Peak memory usage during processing