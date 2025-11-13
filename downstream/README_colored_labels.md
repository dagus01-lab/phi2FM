# Colored Label Images

This document describes the colored label images generated from the numpy label files.

## Generated Structure

For each task, the following structure is created:

```
extracted_samples/
├── {task_name}/
│   ├── img/               # RGB images (.png)
│   ├── label/             # Original one-hot labels (.npy)
│   ├── label_images/      # Colored label images (.png)
│   │   ├── {task}_sample_00000.png
│   │   ├── {task}_sample_01948.png
│   │   └── ...
│   └── {task_name}_legend.png  # NEW: Legend showing class colors
```

## Color Mappings

The color mappings are now unified across all tasks using a consistent palette:

### Universal Color Palette
- **Class 0**: Dark Green (0, 100, 0) - Tree Cover
- **Class 1**: Orange (34, 187, 255) - Shrubland  
- **Class 2**: Yellow (76, 255, 255) - Grassland
- **Class 3**: Purple (255, 150, 240) - Cropland
- **Class 4**: Red (0, 0, 250) - Built-up
- **Class 5**: Gray (180, 180, 180) - Bare/Sparse Vegetation
- **Class 6**: Light Gray (240, 240, 240) - Snow and Ice
- **Class 7**: Blue (200, 100, 0) - Permanent Water
- **Class 8**: Teal (160, 150, 0) - Herbaceous Wetland
- **Class 9**: Bright Green (117, 207, 0) - Mangroves
- **Class 10**: Beige (160, 230, 250) - Moss and Lichen

### Task-Specific Label Mappings

#### Fire Detection (4 classes)
- **Class 0**: Safe → Dark Green
- **Class 1**: Fire → Orange
- **Class 2**: Burnt → Yellow
- **Class 3**: Water → Purple

#### Burned Area (4 classes)
- **Class 0**: Background → Dark Green
- **Class 1**: Burned Area → Orange
- **Class 2**: Clouds → Yellow
- **Class 3**: Waterbodies → Purple

#### Clouds (5 classes)
- **Class 0**: No cloud → Dark Green
- **Class 1**: Cloud → Orange
- **Class 2**: value2 → Yellow
- **Class 3**: value3 → Purple
- **Class 4**: value4 → Red

#### World Floods (3 classes)
- **Class 0**: Clouds → Dark Green
- **Class 1**: Land → Orange
- **Class 2**: Water → Yellow

#### Anomaly Detection (9 classes)
- **Class 0**: NO DATA → Dark Green
- **Class 1**: CLEAR WATER → Orange
- **Class 2**: TURBID WATER → Yellow
- **Class 3**: LAND → Purple
- **Class 4**: PLASTIC → Red
- **Class 5**: OIL → Gray
- **Class 6**: ALGAE → Light Gray
- **Class 7**: SEDIMENTS → Blue
- **Class 8**: CLOUD → Teal

#### Land Cover Classification (11 classes)
- **Class 0**: Tree Cover → Dark Green
- **Class 1**: Shrubland → Orange
- **Class 2**: Grassland → Yellow
- **Class 3**: Cropland → Purple
- **Class 4**: Built-up → Red
- **Class 5**: Bare/Sparse Vegetation → Gray
- **Class 6**: Snow and Ice → Light Gray
- **Class 7**: Permanent Water → Blue
- **Class 8**: Herbaceous Wetland → Teal
- **Class 9**: Mangroves → Bright Green
- **Class 10**: Moss and Lichen → Beige

## Legend Files

Each task now includes a legend file (`{task_name}_legend.png`) that shows:
- Class indices (0, 1, 2, ...)
- Class names (from the TASKS dictionary)
- Corresponding color patches

### Viewing Legends

```bash
cd /home/gdaga/phi2FM/downstream
conda activate esa-phisatnet
python view_legends.py
```

## Usage Examples

### Load and Display Colored Labels

```python
import cv2
import matplotlib.pyplot as plt

# Load colored label image
colored_label = cv2.imread('extracted_samples/clouds/label_images/clouds_sample_00000.png')
colored_label = cv2.cvtColor(colored_label, cv2.COLOR_BGR2RGB)

# Display
plt.imshow(colored_label)
plt.title('Colored Label Image')
plt.axis('off')
plt.show()
```

### Compare with Original Data

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load all three types
rgb_img = cv2.imread('extracted_samples/clouds/img/clouds_sample_00000.png')
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

colored_label = cv2.imread('extracted_samples/clouds/label_images/clouds_sample_00000.png')
colored_label = cv2.cvtColor(colored_label, cv2.COLOR_BGR2RGB)

original_label = np.load('extracted_samples/clouds/label/clouds_sample_00000.npy')

# Display side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(rgb_img)
ax1.set_title('RGB Image')
ax1.axis('off')

ax2.imshow(colored_label)
ax2.set_title('Colored Label')
ax2.axis('off')

# Show class indices from original
if original_label.ndim == 3:
    class_map = np.argmax(original_label, axis=0)
    ax3.imshow(class_map, cmap='tab10')
    ax3.set_title('Class Indices')
else:
    class_idx = np.argmax(original_label)
    ax3.text(0.5, 0.5, f'Class: {class_idx}', ha='center', va='center')
    ax3.set_title('Classification')

plt.tight_layout()
plt.show()
```

### Use the Viewer Script

```bash
cd /home/gdaga/phi2FM/downstream
conda activate esa-phisatnet
python view_colored_labels.py
```

## File Counts by Task

- **burned_area**: 195 colored label images
- **clouds**: 100 colored label images  
- **worldfloods**: 199 colored label images
- **fire**: 190 colored label images
- **anomaly_detection**: 194 colored label images

## Notes

1. **Segmentation tasks**: Each pixel is colored according to its class
2. **Classification tasks**: A solid colored 100×100 square represents the predicted class
3. **One-hot conversion**: Original one-hot labels are converted to class indices using `np.argmax()`
4. **Color format**: Images are saved in BGR format (OpenCV default) but displayed in RGB

## Regenerating Colors

To change the color mappings, edit the `COLOR_MAPS` dictionary in `create_colored_labels.py`:

```python
COLOR_MAPS = {
    "clouds": {
        0: (255, 0, 0),      # Blue (No cloud) -> Change to your preferred BGR color
        1: (255, 255, 255),  # White (Cloud)
        # ... etc
    }
}
```

Then run:
```bash
python create_colored_labels.py
```