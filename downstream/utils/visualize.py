import random
import time

import torch

# random.seed(time.time())

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import buteo as beo
from matplotlib import gridspec

from utils.data_protocol import protocol_fewshot
from utils import config_lc
from utils import config_kg

from utils import Prithvi_100M_config

# statistics used to normalize images before passing to the model
MEANS_PRITHVI = np.array(Prithvi_100M_config.data_mean).reshape(1, 1, -1)
STDS_PRITHVI = np.array(Prithvi_100M_config.data_std).reshape(1, 1, -1)

def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def render_s2_as_rgb(arr, channel_first=False):
    # If there are nodata values, lets cast them to zero.
    if np.ma.isMaskedArray(arr):
        arr = np.ma.getdata(arr.filled(0))

    if channel_first:
        arr = beo.channel_first_to_last(arr)

    if arr.shape[-1] == 6:
        arr = (arr * STDS_PRITHVI) + MEANS_PRITHVI
        np.divide(arr, 10000.0, out=arr)

    # Select only Blue, green, and red. Then invert the order to have R-G-B
    rgb_slice = arr[:, :, 0:3][:, :, ::-1]

    # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
    # Which produces dark images.
    rgb_slice = np.clip(
        rgb_slice,
        np.quantile(rgb_slice, 0.02),
        np.quantile(rgb_slice, 0.98),
    )

    # The current slice is uint16, but we want an uint8 RGB render.
    # We normalise the layer by dividing with the maximum value in the image.
    # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
    rgb_slice = (rgb_slice / rgb_slice.max()) * 255.0

    # We then round to the nearest integer and cast it to uint8.
    rgb_slice = np.rint(rgb_slice).astype(np.uint8)

    return rgb_slice

def decode_date(encoded_date):
    doy_sin, doy_cos = encoded_date

    doy = np.arctan2((2 * doy_sin - 1), (2 * doy_cos - 1)) * 365 / (2 * np.pi)

    if doy < 1:
        doy += 365

    return np.array([np.round(doy)])


def decode_coordinates(encoded_coords):
    lat_enc, long_sin, long_cos = encoded_coords

    lat = -lat_enc * 180 + 90

    long = np.arctan2((2 * long_sin - 1), (2 * long_cos - 1)) * 360 / (2 * np.pi)

    return np.array([lat, long])


def encode_coordinates(coords):
    lat, long = coords

    lat = (-lat + 90) / 180

    long_sin = (np.sin(long * 2 * np.pi / 360) + 1) / 2

    long_cos = (np.cos(long * 2 * np.pi / 360) + 1) / 2

    return np.array([lat, long_sin, long_cos], dtype=np.float32)


def visualize(x, y, y_pred=None, images=5, channel_first=False, vmin=0, vmax=1, save_path=None):
    if images > x.shape[0]:
        images = x.shape[0]

    rows = images
    if y_pred is None:
        columns = 2
    else:
        columns = 3
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    indexes = random.sample(range(0, x.shape[0]), images)
    for idx in indexes:
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(y[idx]), vmin=vmin, vmax=vmax, cmap='magma')
        plt.axis('on')
        plt.grid()

        if y_pred is not None:
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(np.squeeze(y_pred[idx]), vmin=vmin, vmax=vmax, cmap='magma')
            plt.axis('on')
            plt.grid()

    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

















import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from collections import Counter

color_map = {
    0: (0, 0, 255),       # Water / Nodata
    1: (0, 128, 0),       # Af Tropical, Rainforest
    2: (34, 139, 34),     # Am Tropical, Monsoon
    3: (60, 179, 113),    # Aw Tropical, Savannah
    4: (205, 133, 63),    # Bwh Arid, Desert, Hot
    5: (210, 180, 140),   # Bwk Arid, Desert, Cold
    6: (188, 143, 143),   # Bsh Arid, Steppe, Hot
    7: (218, 165, 32),    # Bsk Arid, Steppe, Cold
    8: (255, 0, 0),       # Csa Temperate, Dry Summer, Hot Summer
    9: (255, 69, 0),      # Csb Temperate, Dry Summer, Warm Summer
    10: (255, 140, 0),    # Csc Temperate, Dry Summer, Cold Summer
    11: (255, 215, 0),    # Cwa Temperate, Dry Winter, Hot Summer
    12: (255, 255, 0),    # Cwb Temperate, Dry Winter, Warm Summer
    13: (154, 205, 50),   # Cwc Temperate, Dry Winter, Cold Summer
    14: (173, 255, 47),   # Cfa Temperate, No Dry Season, Hot Summer
    15: (124, 252, 0),    # Cfb Temperate, No Dry Season, Warm Summer
    16: (0, 255, 127),    # Cfc Temperate, No Dry Season, Cold Summer
    17: (0, 255, 255),    # Dsa Cold, Dry Summer, Hot Summer
    18: (70, 130, 180),   # Dsb Cold, Dry Summer, Warm Summer
    19: (100, 149, 237),  # Dsc Cold, Dry Summer, Cold Summer
    20: (30, 144, 255),   # Dsd Cold, Dry Summer, Very Cold Winter
    21: (135, 206, 250),  # Dwa Cold, Dry Winter, Hot Summer
    22: (70, 130, 180),   # Dwb Cold, Dry Winter, Warm Summer
    23: (135, 206, 235),  # Dwc Cold, Dry Winter, Cold Summer
    24: (0, 191, 255),    # Dwd Cold, Dry Winter, Very Cold Winter
    25: (176, 224, 230),  # Dfa Cold, No Dry Season, Hot Summer
    26: (173, 216, 230),  # Dfb Cold, No Dry Season, Warm Summer
    27: (176, 196, 222),  # Dfc Cold, No Dry Season, Cold Summer
    28: (176, 196, 222),  # Dfd Cold, No Dry Season, Very Cold Winter
    29: (255, 250, 250),  # Et Polar, Tundra
    30: (245, 245, 245)   # Ef Polar, Frost
}

climate_map = {
    0: 'Water / Nodata',
    1: 'Af Tropical, Rainforest',
    2: 'Am Tropical, Monsoon',
    3: 'Aw Tropical, Savannah',
    4: 'Bwh Arid, Desert, Hot',
    5: 'Bwk Arid, Desert, Cold',
    6: 'Bsh Arid, Steppe, Hot',
    7: 'Bsk Arid, Steppe, Cold',
    8: 'Csa Temperate',
    9: 'Csb Temperate',
    10: 'Csc Temperate',
    11: 'Cwa Temperate',
    12: 'Cwb Temperate',
    13: 'Cwc Temperate',
    14: 'Cfa Temperate',
    15: 'Cfb Temperate',
    16: 'Cfc Temperate',
    17: 'Dsa Cold',
    18: 'Dsb Cold',
    19: 'Dsc Cold',
    20: 'Dsd Cold',
    21: 'Dwa Cold',
    22: 'Dwb Cold',
    23: 'Dwc Cold',
    24: 'Dwd Cold',
    25: 'Dfa Cold',
    26: 'Dfb Cold',
    27: 'Dfc Cold',
    28: 'Dfd Cold',
    29: 'Et Polar, Tundra',
    30: 'Ef Polar, Frost'
}

reduced_color_map = {
    0: (0, 0, 255),       # Water / Nodata
    1: (255, 120, 0),     # Tropical
    2: (210, 190, 175),   # Arid
    3: (0, 100, 0),    # Temperate
    4: (176, 224, 230),   # Cold
    5: (255, 250, 250),   # Polar
}

reduced_climate_map = {
    0: 'Water / Nodata',
    1: 'Tropical',
    2: 'Arid',
    3: 'Temperate',
    4: 'Cold',
    5: 'Polar',
}

# color_map = reduced_color_map
# climate_map = reduced_climate_map

def get_climate_rgb_and_patches(climate_curr, max_patches=5):
    """
    Converts climate indices to an RGB image and creates legend patches.
    Limits the number of patches to `max_patches`, selecting the most frequent classes.
    """
    # When a single value is passed for climate_curr, 
    # it may be a scalar. Treat it as a single-pixel image.
    if np.isscalar(climate_curr):
        climate_curr = np.array([[climate_curr]])
    
    # Flatten the climate array to count frequencies
    flat_climate = climate_curr.flatten()
    climate_counts = Counter(flat_climate)
    
    # Get the most common climate classes up to max_patches
    most_common = climate_counts.most_common(max_patches)
    selected_classes = [cls for cls, _ in most_common]
    
    # Convert selected climate indices to RGB
    H, W = climate_curr.shape
    climate_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c_idx in selected_classes:
        if c_idx in color_map:
            climate_rgb[climate_curr == c_idx] = color_map[c_idx]
    
    # Create legend patches
    patches = []
    for c in selected_classes:
        if c in color_map:
            color_norm = np.array(color_map[c]) / 255.0
            patches.append(Patch(color=color_norm, label=f"{c}: {climate_map[c]}"))
    
    return climate_rgb, patches




def visualize_pretrain(
        x, y, y_pred=None, images=5, save_path=None, 
        apply_zoom=False, fixed_task=None, climate_segm=False):
    """
    Visualize pretraining outputs. If `climate_segm` is True, 
    show segmentation maps for climate. Otherwise, show a 
    classification table for climate (label vs. prediction).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from scipy.special import softmax
    from collections import Counter

    # Example color_map and climate_map for reference
    # These should exist in your global scope or be imported
    # color_map   = {0: (255,   0,   0),
    #                1: (  0, 255,   0),
    #                2: (  0,   0, 255)}
    #
    # climate_map = {0: "TROPICAL",
    #                1: "ARID",
    #                2: "TEMPERATE"}
    
    # ---------------------------------------
    # Number of images to visualize
    # ---------------------------------------
    if images > x.shape[0]:
        images = x.shape[0]

    # Example means/std
    MEANS_MAJORTOM = np.array([0., 0., 0., 0., 0., 0., 0., 0.]) 
    STD_MAJORTOM   = np.array([1., 1., 1., 1., 1., 1., 1., 1.])  

    # Move to CPU and convert to numpy
    x = x.cpu().detach().numpy()
    y = {k: v.cpu().detach().numpy() for k, v in y.items()}

    if y_pred is not None:
        if fixed_task is None:
            y_pred = {k: v.cpu().detach().numpy() for k, v in y_pred.items()}
        else:
            y_pred = {fixed_task: y_pred[fixed_task].cpu().detach().numpy()}

    # ---------------------------------------
    # Denormalize Corrupted Images
    # ---------------------------------------
    corrupted_images = x[:images]
    corrupted_images = (corrupted_images * STD_MAJORTOM.reshape(1, -1, 1, 1)) \
                       + MEANS_MAJORTOM.reshape(1, -1, 1, 1)
    # Convert to RGB order
    rgb_corrupted_images = corrupted_images[:, [2, 1, 0], :, :]

    # ---------------------------------------
    # Handle tasks
    # ---------------------------------------
    # 1) Coordinates
    if fixed_task is None or fixed_task == 'coords':
        coords_pred = y_pred['coords'][:images]
        coords = y['coords'][:images]
    else:
        coords_pred = np.zeros((images,4))
        coords = np.zeros((images,4))

    # 2) Climate
    if fixed_task is None or fixed_task == 'climate':
        climate = y['climate'][:images]
        climate_logits = y_pred['climate'][:images]  # shape (N, #classes) or (N,H,W)
        
        if climate_segm:
            # For segmentation, we typically have shape (N, #classes, H, W)
            # Convert to predicted indices
            climate_probs = softmax(climate_logits, axis=1)  # shape (N, #classes, H, W)
            climate_pred = np.argmax(climate_probs, axis=1)  # shape (N, H, W)
        else:
            # For classification, we have shape (N, #classes)
            climate_probs = softmax(climate_logits, axis=1)    # shape (N, #classes)
            climate_pred = np.argmax(climate_probs, axis=1)    # shape (N,)
    else:
        # Just create zeros if climate is not used
        climate = np.zeros((images, x.shape[2], x.shape[3]))
        climate_pred = np.zeros((images, x.shape[2], x.shape[3]))
        climate_segm = True  # So that the plotting code doesn't break

    # 3) Zoom factor (conditional)
    if apply_zoom:
        zoom_factor     = y['zoom_factor'][:images]
        zoom_factor_pred = y_pred['zoom_factor'][:images]

    # 4) Reconstruction
    if fixed_task is None or fixed_task == 'reconstruction':
        reconstructed     = y['reconstruction'][:images]
        reconstructed_pred = y_pred['reconstruction'][:images]

        # Denormalize
        reconstructed = (reconstructed * STD_MAJORTOM.reshape(1, -1, 1, 1)) \
                        + MEANS_MAJORTOM.reshape(1, -1, 1, 1)
        reconstructed_pred = (reconstructed_pred * STD_MAJORTOM.reshape(1, -1, 1, 1)) \
                             + MEANS_MAJORTOM.reshape(1, -1, 1, 1)
        # Convert to RGB
        reconstructed = reconstructed[:, [2, 1, 0], :, :]
        reconstructed_pred = reconstructed_pred[:, [2, 1, 0], :, :]

        # Optional brightness adjustment for visualization
        for i in range(images):
            mean_brightness = reconstructed[i].mean()
            if mean_brightness < 0.5:
                brightness_factor = 0.5 / (mean_brightness + 1e-6)
            else:
                brightness_factor = 1.0
            rgb_corrupted_images[i] = np.clip(rgb_corrupted_images[i] * brightness_factor, 0, 1)
            reconstructed_pred[i]   = np.clip(reconstructed_pred[i]   * brightness_factor, 0, 1)
            reconstructed[i]        = np.clip(reconstructed[i]        * brightness_factor, 0, 1)
    else:
        reconstructed     = np.zeros((images, 3, x.shape[2], x.shape[3]))
        reconstructed_pred = np.zeros((images, 3, x.shape[2], x.shape[3]))

    # ---------------------------------------
    # Create figure
    # ---------------------------------------
    fig = plt.figure(figsize=(15, 8 * images))
    # Each image block has 3 rows in the GridSpec:
    #   - Row 0: Reconstructed, Climate (or classification table)
    #   - Row 1: Reconstructed Pred, Climate Pred (or classification table)
    #   - Row 2: Table for coords/zoom
    gs = gridspec.GridSpec(3 * images, 3, height_ratios=[1, 1, 0.2]*images, width_ratios=[1.5, 1, 2])

    for i in range(images):
        start_row = 3 * i

        # --------------------
        # 1) Corrupted Image (spans first column, two rows)
        # --------------------
        ax_corrupted = plt.subplot(gs[start_row:start_row+2, 0])
        corrupted_img = np.transpose(rgb_corrupted_images[i], (1, 2, 0))
        ax_corrupted.imshow(corrupted_img)
        ax_corrupted.set_title(f"Image {i+1}", fontsize=16, fontweight='bold')
        ax_corrupted.axis('off')

        # --------------------
        # 2) Reconstructed (top row, col 1)
        # --------------------
        ax_reconstructed = plt.subplot(gs[start_row, 1])
        reconstructed_img = np.transpose(reconstructed[i], (1, 2, 0))
        ax_reconstructed.imshow(reconstructed_img)
        ax_reconstructed.set_title("Reconstructed Label", fontsize=12)
        ax_reconstructed.axis('off')

        # 3) Reconstructed Pred (2nd row, col 1)
        ax_reconstructed_pred = plt.subplot(gs[start_row+1, 1])
        reconstructed_pred_img = np.transpose(reconstructed_pred[i], (1, 2, 0))
        ax_reconstructed_pred.imshow(reconstructed_pred_img)
        ax_reconstructed_pred.set_title("Reconstructed Pred", fontsize=12)
        ax_reconstructed_pred.axis('off')

        # --------------------
        # 4) Climate
        # --------------------
        climate_current = climate[i]
        # For classification, climate[i] is presumably a single integer label
        # For segmentation, climate[i] is (H, W)

        # We'll use an if-else for segmentation vs. classification
        if climate_segm:
            # -- Segmentation Mode --
            climate_pred_current = climate_pred[i]
            # Climate Ground Truth
            ax_climate = plt.subplot(gs[start_row, 2])
            climate_rgb, patches = get_climate_rgb_and_patches(climate_current, max_patches=5)
            ax_climate.imshow(climate_rgb)
            ax_climate.set_title("Climate")
            ax_climate.axis('off')
            if patches:
                ax_climate.legend(handles=patches, bbox_to_anchor=(1.05, 1),
                                  loc='upper left', borderaxespad=0., fontsize=9)

            # Climate Prediction
            ax_climate_pred = plt.subplot(gs[start_row+1, 2])
            climate_pred_rgb, patches_pred = get_climate_rgb_and_patches(climate_pred_current, max_patches=5)
            ax_climate_pred.imshow(climate_pred_rgb)
            ax_climate_pred.set_title("Climate Pred")
            ax_climate_pred.axis('off')
            if patches_pred:
                ax_climate_pred.legend(handles=patches_pred, bbox_to_anchor=(1.05, 1),
                                       loc='upper left', borderaxespad=0., fontsize=9)

        else:
            # -- Classification Mode --
            # Create ONE subplot spanning the two "climate" rows
            ax_climate_class_table = plt.subplot(gs[start_row, 2])
            ax_climate_class_table.axis('off')

            # climate_current is the integer label
            true_label_int = int(climate_current)
            # climate_probs[i] has shape (#classes,) for classification
            sorted_indices = np.argsort(-climate_probs[i])  # sort descending by probability
            top_k = min(3, len(sorted_indices))             # get up to 3 most probable
            top_indices = sorted_indices[:top_k]

            # Retrieve the ground-truth color (0..255 -> normalized)
            true_color = np.array(color_map[true_label_int]) / 255.0
            true_name  = climate_map[true_label_int]

            # Build table text (5 columns): [Title/Type, Color, ID, Name, Probability]
            # 1) First row: a single 'Climate' title
            # 2) Second row: the "Label" row
            # 3) Next rows: top K predictions

            cell_text = []
            cell_colors = []

            # --- Row 0: "Climate" title ---
            cell_text.append(["Climate", "", "", "", ""])
            cell_colors.append(["#f0f0f0"] * 5)  # lightly shade the title row

            # --- Row 1: Label row ---
            cell_text.append([
                "Label",         # first column
                "",             # color column (we fill via cellColours)
                f"#{true_label_int}",  # climate ID
                f"{true_name}",         # climate name
                ""              # probability is empty for the label
            ])
            
            cell_color = 'lightgreen' if true_label_int == top_indices[0] else 'khaki' if true_label_int in top_indices else 'tomato'
            
            cell_colors.append([
                cell_color,
                true_color,  # color cell for the label
                cell_color,
                cell_color,
                cell_color
            ])

            # --- Rows 2..2+K-1: Top K predictions ---
            for rank, c_idx in enumerate(top_indices):
                pred_color = np.array(color_map[c_idx]) / 255.0
                pred_name  = climate_map[c_idx]
                pred_prob  = climate_probs[i, c_idx] * 100

                cell_text.append([
                    f"Pred #{rank+1}",
                    "",
                    f"#{c_idx}",
                    f"{pred_name}",
                    f"{pred_prob:.2f}%"
                ])
                
                correct_color = 'lightgreen' if rank == 0 else 'khaki'
                cell_color = 'white' if c_idx != true_label_int else correct_color
                
                cell_colors.append([
                    cell_color,
                    pred_color,  # color cell
                    cell_color,
                    cell_color,
                    cell_color
                ])

            # Create the single table
            table = ax_climate_class_table.table(
                cellText=cell_text,
                cellColours=cell_colors,
                loc='center',
                cellLoc='center',
                colWidths=[0.2, 0.1, 0.1, 0.45, 0.15]
            )

            # Make the table a bit bigger / more readable
            table.scale(1.0, 2.0)
            table.auto_set_font_size(False)
            table.set_fontsize(12)

            # Make the top (title) row bold
            for col in range(5):
                table[(0, col)].set_text_props(weight='bold')
                table[(1, col)].set_text_props(weight='bold')  # label row bold
                if len(top_indices) > 0:
                    table[(2, col)].set_text_props(weight='bold')  # first prediction row bold



        # --------------------
        # 5) Table (coords, zoom, etc.)
        # --------------------
        if climate_segm:
            ax_table = plt.subplot(gs[start_row+2, :])
        else:
            ax_table = plt.subplot(gs[start_row+1, 2])
        ax_table.axis('off')

        # Format each coordinate value separately into a list of strings
        coords_formatted      = [f"{coord:.4f}" for coord in coords[i]]
        coords_pred_formatted = [f"{coord:.4f}" for coord in coords_pred[i]]

        if apply_zoom:
            zoom_formatted      = f"{zoom_factor[i][0]:.4f}"
            zoom_pred_formatted = f"{zoom_factor_pred[i][0]:.4f}"

            # Append zoom values to the coordinate lists
            row1 = coords_formatted + [zoom_formatted]
            row2 = coords_pred_formatted + [zoom_pred_formatted]

            # Define column headers including Zoom
            col_labels = ['sin(lat)', 'cos(lat)', 'sin(lon)', 'cos(lon)', 'Zoom']
        else:
            # Without zoom, keep just the coordinate values
            row1 = coords_formatted
            row2 = coords_pred_formatted

            # Define column headers without Zoom
            col_labels = ['sin(lat)', 'cos(lat)', 'sin(lon)', 'cos(lon)']

        # Row labels remain the same
        row_labels = ['Label', 'Pred']

        # Create cell text from the two rows
        cell_text = [row1, row2]

        # Create the table with the specified row and column labels
        table = ax_table.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # Highlight headers: bold text and grey background for header cells
        for (row, col), cell in table.get_celld().items():
            if row == 0 and col != -1:   # column headers
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
            if col == -1 and row != -1:  # row headers
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')

    plt.tight_layout()

    # Optionally save
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



import torch
from collections import defaultdict
from tabulate import tabulate

def param_type_to_category(pname: str):
    """
    Map parameter type to 'weight' or 'bias' bucket.
    E.g. 'weight', 'x_skipscale', 'y_skipscale' -> 'weight'
         'bias', 'x_skipbias', 'y_skipbias' -> 'bias'
    """
    if pname.endswith('bias') or pname == 'bias':
        return 'bias'
    else:
        return 'weight'

def parse_name(name: str):
    """
    Parse something like:
        'module.encoder.blocks_down.3.1.norm1.weight'
    into:
        module='encoder',
        layer='blocks_down.3.1',
        sub='norm1',
        param='weight'  (which we will further map to 'weight'/'bias')
    """
    parts = name.split('.')
    # Usually parts[0] == "module", and parts[1] is the high-level module name (e.g. encoder, stem, decoder, etc.)
    module_name = parts[1] if len(parts) > 1 else "unknown"
    # Everything from parts[2:-1] is the "middle" portion
    middle = parts[2:-1]  # e.g. ["blocks_down", "3", "1", "norm1"]
    ptype = parts[-1]     # e.g. "weight", "bias", "x_skipscale", etc.

    if len(middle) == 0:
        # no middle: param is directly under the module?
        # e.g. "module.head_geo.0.weight" => middle=["0"]
        layer = ""
        sub = ""
    elif len(middle) == 1:
        # only one item in the middle => treat that as the "layer"
        layer = middle[0]
        sub = ""
    else:
        # more than one => last part is sub-layer, the rest is the layer
        layer = '.'.join(middle[:-1])
        sub = middle[-1]

    return module_name, layer, sub, ptype


def collect_model_stats(model: torch.nn.Module):
    """
    Collect gradient norms and stats (mean, std, min, max) for weights & biases,
    grouped by (module, layer, sub-layer).
    Returns a nested dictionary.
    """
    data_dict = defaultdict(lambda: {
        'weight_grad': None,   # gradient norm for weight-like param
        'bias_grad': None,     # gradient norm for bias-like param
        'weight_mean': None, 'weight_std': None,  # stats for weight-like
        'weight_min': None, 'weight_max': None,
        'bias_mean': None,   'bias_std': None,    # stats for bias-like
        'bias_min': None,    'bias_max': None
    })

    for name, param in model.named_parameters():
        module_name, layer_name, sub_name, ptype = parse_name(name)
        category = param_type_to_category(ptype)  # 'weight' or 'bias'

        # Calculate gradient norm if grad is not None
        grad_norm = None
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()

        # Gather stats from param.data
        p_mean = param.data.mean().item()
        p_std  = param.data.std().item()
        p_min  = param.data.min().item()
        p_max  = param.data.max().item()

        key = (module_name, layer_name, sub_name)  # used as dict key

        if category == 'weight':
            data_dict[key]['weight_grad'] = grad_norm
            data_dict[key]['weight_mean'] = p_mean
            data_dict[key]['weight_std']  = p_std
            data_dict[key]['weight_min']  = p_min
            data_dict[key]['weight_max']  = p_max
        else:  # 'bias'
            data_dict[key]['bias_grad'] = grad_norm
            data_dict[key]['bias_mean'] = p_mean
            data_dict[key]['bias_std']  = p_std
            data_dict[key]['bias_min']  = p_min
            data_dict[key]['bias_max']  = p_max

    return data_dict


def print_stats_table(stats_dict):
    """
    Turn the stats_dict into a tabular form using tabulate.
    """
    headers = [
        "Module",
        "Layer",
        "Sub-layer",
        "GradNorm (weight)",
        "GradNorm (bias)",
        "Mean ± Std (weight)",
        "Mean ± Std (bias)",
        "Min - Max (weight)",
        "Min - Max (bias)"
    ]

    rows = []
    # Sort the keys so the table has a stable order
    for (module_name, layer_name, sub_name) in sorted(stats_dict.keys()):
        st = stats_dict[(module_name, layer_name, sub_name)]

        # For formatting, handle None or numeric
        w_grad_str = f"{st['weight_grad']:.2e}" if st['weight_grad'] is not None else "None"
        b_grad_str = f"{st['bias_grad']:.2e}"   if st['bias_grad'] is not None   else "None"

        # Mean ± Std
        w_mean_std_str = (
            f"{st['weight_mean']:.2e} ± {st['weight_std']:.2e}"
            if st['weight_mean'] is not None
            else "None"
        )
        b_mean_std_str = (
            f"{st['bias_mean']:.2e} ± {st['bias_std']:.2e}"
            if st['bias_mean'] is not None
            else "None"
        )

        # Min–Max
        w_minmax_str = (
            f"{st['weight_min']:.2e} - {st['weight_max']:.2e}"
            if st['weight_min'] is not None
            else "None"
        )
        b_minmax_str = (
            f"{st['bias_min']:.2e} - {st['bias_max']:.2e}"
            if st['bias_min'] is not None
            else "None"
        )

        rows.append([
            module_name,
            layer_name,
            sub_name,
            w_grad_str,
            b_grad_str,
            w_mean_std_str,
            b_mean_std_str,
            w_minmax_str,
            b_minmax_str
        ])

    print(tabulate(rows, headers=headers, tablefmt="simple", floatfmt=".2e"))

# ------------------------------------------------------------------
# USAGE EXAMPLE (assuming 'model' is your model):
# 1. Forward pass
# 2. Loss calculation
# 3. loss.backward()
# 4. Then collect and print:

# stats = collect_model_stats(model)
# print_stats_table(stats)


def tabulate_losses(train_log_loss, val_log_loss, climate_segm, perceptual_loss):

    # Helper function for formatting numbers in scientific notation
    def sci(x):
        return f"{x:.3e}"

    # Define headers and dynamically exclude "total_variation" if climate_segm is False
    headers = ["", "reconstruction", "perceptual", "climate", "geolocation"]

    if perceptual_loss:
        headers.insert(2, "perceptual")

    if climate_segm:
        headers.insert(-1, "total_variation")

    # Helper function to construct rows conditionally
    def construct_rows(log_loss):
        rows = [
            [f"Total Loss = {sci(log_loss['total_loss'])}",
             sci(log_loss['loss_components']['reconstruction']),
             sci(log_loss['loss_components']['perceptual']) if perceptual_loss else "",
             sci(log_loss['loss_components']['climate'])] +
            ([sci(log_loss['loss_components']['total_variation'])] if climate_segm else []) +
            [sci(log_loss['loss_components']['geolocation'])],

            ["log_sigma",
             sci(log_loss['log_sigmas']['log_sigma_recon']),
             sci(log_loss['log_sigmas']['log_sigma_perc']) if perceptual_loss else "",
             sci(log_loss['log_sigmas']['log_sigma_clim'])] +
            ([sci(log_loss['log_sigmas']['log_sigma_tv'])] if climate_segm else []) +
            [sci(log_loss['log_sigmas']['log_sigma_geo'])],

            ["scaled_loss",
             sci(log_loss['scaled_loss']['reconstruction']),
             sci(log_loss['scaled_loss']['perceptual']) if perceptual_loss else "",
             sci(log_loss['scaled_loss']['climate'])] +
            ([sci(log_loss['scaled_loss']['total_variation'])] if climate_segm else []) +
            [sci(log_loss['scaled_loss']['geolocation'])]
        ]
        return rows

    # Construct rows for training and validation data
    train_rows = construct_rows(train_log_loss)
    val_rows = construct_rows(val_log_loss)

    # Print the tables
    print("Train Loss Table:")
    print(tabulate(train_rows, headers=headers, tablefmt="simple", floatfmt=".3e"))
    print("\nValidation Loss Table:")
    print(tabulate(val_rows, headers=headers, tablefmt="simple", floatfmt=".3e"))


















def visualize_lc(x, y, y_pred=None, images=5, channel_first=False, vmin=0,save_path=None):
    lc_map_names = config_lc.lc_raw_classes
    lc_map = config_lc.lc_model_map
    lc_map_inverted = {v: k for k, v in zip(lc_map.keys(), lc_map.values())}
    vmax = len(lc_map)

    if images > x.shape[0]:
        images = x.shape[0]
        
    # d = 1 if channel_first else -1
    # # y= y.argmax(axis=d)
    # if y_pred is not None:
    #     y_pred = y_pred.argmax(axis=d)
    cmap = (matplotlib.colors.ListedColormap(config_lc.lc_color_map.values()))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    rows = images
    if y_pred is None:
        columns = 2
    else:
        columns = 3
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    indexes = random.sample(range(0, x.shape[0]), images)
    for idx in indexes:
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(y[idx]), vmin=vmin, vmax=vmax, cmap=cmap)
        patches = [mpatches.Patch(color=cmap(norm(u)), label=lc_map_names[lc_map_inverted[u]]) for u in np.unique(y[idx])]
        plt.legend(handles=patches)
        plt.axis('on')
        plt.grid()

        if y_pred is not None:
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(np.squeeze(y_pred[idx]), vmin=vmin, vmax=vmax, cmap=cmap)
            patches = [mpatches.Patch(color=cmap(norm(u)), label=lc_map_names[lc_map_inverted[u]]) for u in np.unique(y_pred[idx])]
            plt.legend(handles=patches)
            plt.axis('on')
            plt.grid()

    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def visualize_building_classification(x, y, y_pred=None, images=5, channel_first=False, num_classes=11, labels=None, save_path=None):

    if images > x.shape[0]:
        images = x.shape[0]

    rows = images
    columns = 1
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    indexes = random.sample(range(0, x.shape[0]), images)
    for idx in indexes:
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        label = y[idx]
        pred = softmax(y_pred[idx])

        max_class = np.argmax(label)
        max_class_pred = np.argmax(pred)

        s1 = (f"Label: Class = {labels[max_class]} "
               f"\n Percentage = {label[max_class]} ")

        s2 = (f"Prediction: Class = {labels[max_class_pred]} "
               f"\n Percentage = {pred[max_class_pred]} ")

        plt.text(25, 25, s1, fontsize=18, bbox=dict(fill=True))

        plt.text(25, 65, s2, fontsize=18, bbox=dict(fill=True))


    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def visualize_lc_classification(
    x, 
    y, 
    y_pred=None, 
    images=5, 
    channel_first=False, 
    num_classes=11, 
    labels=None, 
    save_path=None, 
    threshold=0.5
):
    """
    Visualizes multi-label classification results with a table annotation.

    Parameters:
    - x (np.ndarray or torch.Tensor): Input images.
    - y (np.ndarray or torch.Tensor): Ground truth labels (binary vectors).
    - y_pred (np.ndarray or torch.Tensor, optional): Predicted logits or probabilities.
    - images (int): Number of images to display.
    - channel_first (bool): If True, assumes image data is in channel-first format.
    - num_classes (int): Number of classes.
    - labels (list or None): List of class names.
    - save_path (str or None): Path to save the visualization.
    - threshold (float): Threshold to decide if a class is predicted.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import torch

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def render_s2_as_rgb(arr, channel_first):
        """
        Convert Sentinel-2 data to RGB by selecting and reordering channels.

        Parameters:
        - arr (np.ndarray): Image array.
        - channel_first (bool): If True, the channel dimension is first.

        Returns:
        - rgb (np.ndarray): RGB image array with shape (height, width, 3).
        """
        if channel_first:
            # Original shape: (channels, height, width)
            # Transpose to (height, width, channels)
            arr = np.transpose(arr, (1, 2, 0))
        
        if arr.shape[2] < 3:
            raise ValueError(f"Expected at least 3 channels, but got {arr.shape[2]} channels.")
        
        rgb_channels = arr[:, :, [2, 1, 0]]  # Select channels 2, 1, 0
        rgb_normalized = (rgb_channels - rgb_channels.min()) / (rgb_channels.max() - rgb_channels.min())
        return rgb_normalized

    if images > x.shape[0]:
        images = x.shape[0]

    rows = images
    columns = 1
    fig = plt.figure(figsize=(15 * columns, 10 * rows))

    indexes = random.sample(range(0, x.shape[0]), images)
    for i, idx in enumerate(indexes, 1):
        arr = x[idx]
        try:
            rgb_image = render_s2_as_rgb(arr, channel_first)
        except ValueError as ve:
            print(f"Error processing image index {idx}: {ve}")
            continue

        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.grid(False)

        if y_pred is not None:
            if isinstance(y_pred, torch.Tensor):
                pred = sigmoid(y_pred[idx].cpu().detach().numpy())
            else:
                pred = sigmoid(y_pred[idx])
        else:
            pred = np.zeros(num_classes)

        header = [f"{j}\n{labels[j]}" if labels else f"Class\n{j}" for j in range(num_classes)]
        
        true_row = ["X" if y[idx][j] > 0 else "" for j in range(num_classes)]
        pred_row = ["X" if pred[j] > threshold else "" for j in range(num_classes)]

        table_data = [
            [""] + header,
            ["True Labels"] + true_row,
            ["Predictions"] + pred_row
        ]

        the_table = plt.table(
            cellText=table_data,
            colLoc='center',
            cellLoc='center',
            loc='top',
            colWidths=[0.1] * (num_classes + 1)
        )

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.scale(1, 2)

        plt.subplots_adjust(top=0.8)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    del x
    del y
    if y_pred is not None:
        del y_pred



def visualize_vae(images, labels, outputs, num_images=5, channel_first=False,save_path=None ):
    images = images.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    rows = num_images
    columns = 2
    i = 0
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(10 * columns, 10 * rows))
    reconstruction, meta_data, embeddings_ssl = outputs

    images = np.einsum('nchw->nhwc', images)
    reconstruction = np.einsum('nchw->nhwc', reconstruction.detach().cpu().numpy())

    for idx in range(0, num_images):
        arr_x = images[idx]
        arr_y = reconstruction[idx]

        rgb_x = render_s2_as_rgb(arr_x, False)
        rgb_y = render_s2_as_rgb(arr_y, False)

        kg_label = labels[idx, :31]
        co_ordinate_labels = labels[idx, 31:34]
        time_labels = labels[idx, 34:36]

        coord_out  = meta_data[0][idx]
        time_out = meta_data[1][idx]
        kg_out = meta_data[2][idx]

        lat, long = decode_coordinates(co_ordinate_labels)
        lat_pred, long_pred = decode_coordinates(coord_out.detach().cpu().numpy())

        doy = decode_date(time_labels)
        doy_pred = decode_date(time_out.detach().cpu().numpy())

        climate = config_kg.kg_map[int(np.argmax([kg_label]))]['climate_class_str']
        climate_pred = config_kg.kg_map[int(np.argmax([kg_out.detach().cpu().numpy()]))]['climate_class_str']

        s1 = (f"Prediction: lat-long = {np.round(lat_pred, 2), np.round(long_pred, 2)} "
              f"\n climate = {climate_pred} "
              f"\n DoY = {doy_pred}")

        s2 = (f"Label: lat-long = {np.format_float_positional(lat, 2), np.format_float_positional(long, 2)} "
              f"\n climate = {climate} "
              f"\n DoY = {doy}")


        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_x)

        plt.text(25, 25, s1, fontsize=18, bbox=dict(fill=True))

        plt.text(25, 65, s2, fontsize=18, bbox=dict(fill=True))

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_y)


    fontsize = 96
    axes[0][0].set_title('image', fontdict={'fontsize': fontsize})
    axes[0][1].set_title('recon.', fontdict={'fontsize': fontsize})

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.clf()
    plt.close()

def visualize_paper():
    vmin = 0


    lc_map_names = config_lc.lc_raw_classes
    lc_map = config_lc.lc_model_map
    lc_map_inverted = {v: k for k, v in zip(lc_map.keys(), lc_map.values())}
    vmax = len(lc_map)

    cmap = (matplotlib.colors.ListedColormap(config_lc.lc_color_map.values()))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


    images = np.load('/phileo_data/downstream/downstream_dataset_patches_np/europe_1_train_s2.npy')
    lc_labels = np.load('/phileo_data/downstream/downstream_dataset_patches_np/europe_1_train_label_lc.npy')
    road_labels = np.load('/phileo_data/downstream/downstream_dataset_patches_np/europe_1_train_label_roads.npy')
    building_labels = np.load('/phileo_data/downstream/downstream_dataset_patches_np/europe_1_train_label_building.npy')

    rows = 25

    columns = 4
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    indexes = random.sample(range(0, images.shape[0]), rows)
    for idx in indexes:
        rgb_image = render_s2_as_rgb(images[idx], channel_first=False)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)

        u, inv = np.unique(lc_labels[idx], return_inverse=True)
        y = np.array([lc_map[x] for x in u])[inv].reshape(lc_labels[idx].shape)
        plt.imshow(np.squeeze(y), vmin=vmin, vmax=vmax, cmap=cmap)
        patches = [mpatches.Patch(color=cmap(norm(lc_map[u])), label=lc_map_names[u]) for u in
                   np.unique(lc_labels[idx])]
        plt.legend(handles=patches)
        plt.axis('on')

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(building_labels[idx]), vmin=vmin, vmax=1, cmap='magma')
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(road_labels[idx]), vmin=vmin, vmax=1, cmap='magma')
        plt.axis('on')
        plt.grid()

    fig.tight_layout()
    plt.savefig('visualization_of_labels.png')
    plt.close()


def visualize_geolocation(x, y, y_pred=None, images=5, channel_first=False, save_path=None):

    if images > x.shape[0]:
        images = x.shape[0]

    rows = images
    columns = 1
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    indexes = random.sample(range(0, x.shape[0]), images)
    for idx in indexes:
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        label = y[idx]
        pred = y_pred[idx]

        s1 = (f"True Location: {label}")

        s2 = (f"Predicted Location: {pred}")

        plt.text(25, 25, s1, fontsize=18, bbox=dict(fill=True))

        plt.text(25, 65, s2, fontsize=18, bbox=dict(fill=True))


    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()




if __name__ == '__main__':
    visualize_paper()