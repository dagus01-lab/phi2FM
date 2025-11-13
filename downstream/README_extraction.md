# Dataset Sample Extraction

This directory contains scripts to extract samples from downstream task datasets for visualization and analysis.

## Files

- `extract_samples_simple.py` - Main extraction script (recommended)
- `extract_dataset_samples.py` - Alternative extraction script with more features
- `README_extraction.md` - This file

## Usage

### Simple Extraction (Recommended)

```bash
cd /home/gdaga/phi2FM/downstream
python extract_samples_simple.py
```

This will extract 50 samples from each task dataset and save them to `/home/gdaga/phi2FM/downstream/extracted_samples/`

### Configuration

You can modify the script configuration by editing the variables at the top of `extract_samples_simple.py`:

```python
# Output directory (change this to your desired location)
OUTPUT_BASE_DIR = "/home/gdaga/phi2FM/downstream/extracted_samples"

# Number of samples to extract per task
NUM_SAMPLES_PER_TASK = 50

# Task configurations
TASKS = {
    "fire": {...},
    "burned_area": {...},
    "clouds": {...},
    "worldfloods": {...}
}
```

## Output Structure

The script creates the following directory structure:

```
extracted_samples/
├── fire/
│   ├── img/                    # RGB images ready for visualization (.png)
│   ├── label/                  # One-hot encoded labels (.npy)
│   ├── label_images/           # Colored label images (.png)
│   ├── comparison_plots/       # NEW: Matplotlib comparison plots (.png)
│   │   ├── fire_comparison_plot_01.png
│   │   ├── fire_comparison_plot_02.png
│   │   └── ...
│   └── fire_legend.png         # Legend showing class colors
├── burned_area/
│   ├── img/
│   ├── label/
│   ├── label_images/
│   ├── comparison_plots/       # NEW: Comparison plots
│   └── burned_area_legend.png
└── ...
```

## Image Processing

- **Images**: Converted to RGB using bands [2,1,0] (pseudo-RGB for satellite data)
- **Normalization**: 2nd-98th percentile normalization
- **Format**: 8-bit PNG files ready for matplotlib visualization
- **Naming**: `{task}_sample_{index:05d}.png`

## Label Processing

- **Format**: One-hot encoded numpy arrays
- **Segmentation tasks**: Shape [C, H, W] where C is number of classes
- **Classification tasks**: Shape [C] where C is number of classes
- **Naming**: `{task}_sample_{index:05d}.npy`

## Colored Label Images

- **Purpose**: Visualization-ready colored versions of labels
- **Colors**: Unified color palette across all tasks (see legend files)
- **Format**: 8-bit PNG files with class-specific colors
- **Naming**: `{task}_sample_{index:05d}.png`

## Comparison Plots (NEW!)

- **Purpose**: Side-by-side visualization of original images and colored labels
- **Format**: High-resolution matplotlib plots (150 DPI)
- **Layout**: 50 image-label pairs per plot, arranged in a 5×10 grid
- **Features**:
  - Legend at the top showing class names and colors
  - Original image and colored label side-by-side for each sample
  - Multiple plots if more than 50 samples (e.g., plot_01.png, plot_02.png)
- **Naming**: `{task}_comparison_plot_{number:02d}.png`

### Comparison Plot Details:
- **Grid Layout**: 10 columns (5 image-label pairs per row)
- **Samples per Plot**: 50 pairs maximum
- **Image Size**: 20×12 inches minimum, adjusts based on number of rows
- **Legend**: Shows all classes with color patches and names
- **Title Format**: "Image XXXXX" and "Label XXXXX" for each pair

## Task Details

### Fire Detection
- **Classes**: 4 (safe, fire, burnt, water)
- **Type**: Classification
- **Dataset**: `/Data/fire_dataset/fire_dataset.zarr`

### Burned Area
- **Classes**: 4 (Background, Burned Area, Clouds, Waterbodies)  
- **Type**: Segmentation
- **Dataset**: `/Data/lpl_burned_area/burned.zarr`

### Cloud Detection
- **Classes**: 5 (No cloud, Cloud, value2, value3, value4)
- **Type**: Segmentation  
- **Dataset**: `/Data/phisatnet_clouds/phisatnet_clouds.zarr`

### World Floods
- **Classes**: 3 (Clouds, Land, Water)
- **Type**: Segmentation
- **Dataset**: `/Data/worldfloods/worldfloods.zarr`

## Loading Extracted Data

## Usage Examples

### Run Full Extraction (includes comparison plots)

```bash
cd /home/gdaga/phi2FM/downstream
conda activate esa-phisatnet
python extract_samples_simple.py
```

### Create Only Comparison Plots (for existing data)

```bash
python create_all_comparison_plots.py
```

### View Individual Components

#### Load an image:
```python
import cv2
import numpy as np

# Load RGB image
img = cv2.imread('extracted_samples/fire/img/fire_sample_00100.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
```

#### Load a label:
```python
import numpy as np

# Load one-hot encoded label
label = np.load('extracted_samples/fire/label/fire_sample_00100.npy')

# For segmentation: label.shape = [C, H, W]
# For classification: label.shape = [C]

# Convert back to class indices (if needed)
if label.ndim > 1:  # Segmentation
    class_map = np.argmax(label, axis=0)  # [H, W]
else:  # Classification
    class_idx = np.argmax(label)  # Single class index
```

#### Load a colored label:
```python
import cv2

# Load colored label image
colored_label = cv2.imread('extracted_samples/fire/label_images/fire_sample_00100.png')
colored_label = cv2.cvtColor(colored_label, cv2.COLOR_BGR2RGB)
```

#### View comparison plots:
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display comparison plot
plot_img = mpimg.imread('extracted_samples/fire/comparison_plots/fire_comparison_plot_01.png')
plt.figure(figsize=(20, 12))
plt.imshow(plot_img)
plt.axis('off')
plt.show()
```

## Troubleshooting

1. **Missing config files**: Make sure the YAML config files exist in the `args/finetune_FMs/` directories
2. **Dataset paths**: Verify that the dataset paths in the TASKS configuration match your actual data locations
3. **Memory issues**: The script loads one sample at a time to minimize memory usage
4. **Permissions**: Make sure you have write permissions to the output directory

## Sample Usage in Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load a sample
task = "fire"
sample_id = "00100"

img = cv2.imread(f'extracted_samples/{task}/img/{task}_sample_{sample_id}.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

label = np.load(f'extracted_samples/{task}/label/{task}_sample_{sample_id}.npy')

# Display
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
# For segmentation tasks
if label.ndim > 1:
    class_map = np.argmax(label, axis=0)
    plt.imshow(class_map, cmap='tab10')
    plt.title('Ground Truth Segmentation')
else:
    # For classification tasks
    class_idx = np.argmax(label)
    plt.bar(range(len(label)), label)
    plt.title(f'Ground Truth: Class {class_idx}')
plt.axis('off')

plt.tight_layout()
plt.show()
```