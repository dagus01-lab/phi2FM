import numpy as np
from osgeo import gdal
import warnings


def src_band(dataset, identifier, warn=True):
    if identifier == 'bands':
        identifier = list(range(1, 9))
    if identifier == 'labels':
        identifier = list(range(9, dataset.RasterCount + 1))
    if identifier == 'all':
        identifier = list(range(1, dataset.RasterCount + 1))
    if identifier == 'rgb':
        return rgb_bands(dataset, normalize=False)
    if identifier == 'rgb_norm':
        return rgb_bands(dataset, normalize=True)
    if identifier == 'bands_names':
        bands = [dataset.GetRasterBand(i).GetDescription() for i in range(1, dataset.RasterCount + 1)]
        datatypes = [gdal.GetDataTypeName(dataset.GetRasterBand(i).DataType) for i in range(1, dataset.RasterCount + 1)]
        bands_dict = dict(zip(bands, datatypes))
        return bands_dict

    # Helper function to get a band by description or index
    def get_band_by_identifier(dataset, ident):
        if isinstance(ident, str):  # Search by description if it's a string                
            for i in range(dataset.RasterCount):
                band_index = i + 1  # GDAL band indices are 1-based
                band = dataset.GetRasterBand(band_index)
                description = band.GetDescription()
                if description == ident:
                    return band.ReadAsArray()  # Return the band data
            if warn:
                warnings.warn(f"Band with description '{ident}' not found.")
            return None
        elif isinstance(ident, int):  # Read directly if it's an integer
            if 1 <= ident <= dataset.RasterCount:  # Ensure index is within valid range
                band = dataset.GetRasterBand(ident)
                return band.ReadAsArray()  # Return the band data
            else:
                if warn:
                    warnings.warn(f"Band index {ident} is out of range.")
                return None
        else:
            if warn:
                warnings.warn(f"Identifier must be a string (description) or an integer (index).")
            return None
    
    # If the identifier is a list or tuple, retrieve multiple bands
    if isinstance(identifier, (list, tuple)):
        bands = [get_band_by_identifier(dataset, ident) for ident in identifier]
        return np.stack(bands, axis=0)  # Stack the bands into a single array
    
    # If the identifier is a single value (string or integer)
    else:
        return get_band_by_identifier(dataset, identifier)

def rgb_bands(dataset, normalize=False):
    # Select the RGB bands by their descriptions
    rgb_image = src_band(dataset, ["PHISAT2-B04", "PHISAT2-B03", "PHISAT2-B02"], warn=False)
    if rgb_image[0] is None:
        print("RGB bands not found. Using default bands 3, 2, 1.")
        rgb_image = src_band(dataset, [4,3,2])  # Indices start at 1 instead of 0
    # Apply normalization if needed
    print(rgb_image.shape)
    if normalize:
        if np.max(rgb_image) > 100:
            rgb_image = np.log1p(rgb_image)
        else:
            rgb_image = np.clip(3.5 * rgb_image, 0, 1)

        for i in range(rgb_image.shape[0]):  # Loop over the bands (axis 0)
            rgb_image[i] = (rgb_image[i] - rgb_image[i].min()) / (rgb_image[i].max() - rgb_image[i].min())
    
    # Transpose the bands to [height, width, channels]
    rgb_array = np.transpose(rgb_image, [1, 2, 0])
    
    return rgb_array
