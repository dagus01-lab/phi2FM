from osgeo import gdal
gdal.DontUseExceptions()
import numpy as np
import os
from tqdm import tqdm


def save_tif(filename, data_arrays, band_names, data_type, geotransform, projection, metadata=None):
    driver = gdal.GetDriverByName('GTiff')
    options = ['COMPRESS=LZW']  # Use LZW compression

    out_ds = driver.Create(
        filename,
        dataset.RasterXSize,
        dataset.RasterYSize,
        len(data_arrays),
        data_type,
        options
    )

    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    # Set metadata if provided
    if metadata:
        out_ds.SetMetadata(metadata)

    for idx, array in enumerate(data_arrays):
        out_band = out_ds.GetRasterBand(idx + 1)
        out_band.WriteArray(array)
        out_band.SetDescription(band_names[idx])

    out_ds.FlushCache()
    out_ds = None  # Close the dataset




# Open the original TIFF file
input_folder = '/home/ccollado/phileo_phisat2/L1A/tiff_files'
output_folder = '/home/ccollado/phileo_phisat2/L1A/converted_tiff_files'
input_files = os.listdir(input_folder)
input_files.sort()
print(f'Converting {len(input_files)} files')

# DICT Min and Max values for each band
overall_stats = {
    'PHISAT2-B02': (-1.8809680938720703, 1128.3280029296875),
    'PHISAT2-B03': (-2.3193440437316895, 934.764404296875),
    'PHISAT2-B04': (-2.3585681915283203, 879.5845336914062),
    'PHISAT2-PAN': (-0.7101852893829346, 863.93017578125),
    'PHISAT2-B08': (-1.511198878288269, 639.3428955078125),
    'PHISAT2-B05': (-3.698456287384033, 909.0509643554688),
    'PHISAT2-B06': (-3.8417840003967285, 892.326416015625),
    'PHISAT2-B07': (-2.4129557609558105, 977.6026611328125)
    }



for input_file in tqdm(input_files):
    
    if os.path.splitext(input_file)[0] + '_phi2.tif' in os.listdir(output_folder):
        continue

    dataset = gdal.Open(os.path.join(input_folder, input_file))

    # Retrieve geospatial information
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Extract metadata from the original dataset
    metadata = dataset.GetMetadata()

    # Initialize dictionaries to store band data
    phisat2_bands = {}
    world_cover = None
    cloud_prob = None
    buildings = None
    roads = None

    # Iterate over each band in the dataset
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        band_name = band.GetDescription()
        data = band.ReadAsArray()
                    
        if band_name.startswith('PHISAT2'):
            min_val, max_val = overall_stats[band_name]

            if min_val == max_val:
                raise ValueError(f"Min and max values are the same for band {band_name}")
            scale_factor = 65534 / (max_val - min_val)
            data_copy = data.copy()

            nan_mask = np.isnan(data_copy)
            data_copy[nan_mask] = min_val

            data_scaled = (data_copy - min_val) * scale_factor
            
            data_scaled = np.clip(data_scaled, 0, 65534)
            data_scaled[nan_mask] = 65535

            data_uint16 = data_scaled.astype(np.uint16)

            if not np.all((data_uint16 >= 0) & (data_uint16 <= 65535)):
                raise ValueError(f'File {input_file} has values outside the range 0 to 65535')
            
            # import pdb; pdb.set_trace() 
            phisat2_bands[band_name] = data_uint16

        elif band_name == 'WORLD_COVER':
            data_uint8 = data.astype(np.uint8)
            world_cover = data_uint8
        elif band_name == 'CLOUD_PROB':
            # Scale probabilities to 0-255 and convert to uint8
            cloud_prob = (data * 255).astype(np.uint8)
        elif band_name == 'BUILDINGS':
            buildings = data
        elif band_name == 'ROADS':
            roads = data

    # Save PHISAT2 Bands with metadata
    phisat2_band_names = list(phisat2_bands.keys())
    phisat2_arrays = [phisat2_bands[name] for name in phisat2_band_names]
    output_filename = os.path.join(output_folder, os.path.splitext(input_file)[0] + '_phi2.tif')
    save_tif(
        filename=output_filename,
        data_arrays=phisat2_arrays,
        band_names=phisat2_band_names,
        data_type=gdal.GDT_UInt16,
        geotransform=geotransform,
        projection=projection,
        metadata=metadata  # Include metadata
    )

    if world_cover is not None:
        output_filename = os.path.join(output_folder, os.path.splitext(input_file)[0] + '_lc.tif')
        save_tif(
            filename=output_filename,
            data_arrays=[world_cover],
            band_names=['WORLD_COVER'],
            data_type=gdal.GDT_Byte,
            geotransform=geotransform,
            projection=projection
        )

    if cloud_prob is not None:
        output_filename = os.path.join(output_folder, os.path.splitext(input_file)[0] + '_cloud.tif')
        save_tif(
            filename=output_filename,
            data_arrays=[cloud_prob],
            band_names=['CLOUD_PROB'],
            data_type=gdal.GDT_Byte,  # Updated data type to Byte
            geotransform=geotransform,
            projection=projection
        )

    if buildings is not None:
        output_filename = os.path.join(output_folder, os.path.splitext(input_file)[0] + '_building.tif')
        save_tif(
            filename=output_filename,
            data_arrays=[buildings],
            band_names=['BUILDINGS'],
            data_type=gdal.GDT_Float32,
            geotransform=geotransform,
            projection=projection
        )

    if roads is not None:
        output_filename = os.path.join(output_folder, os.path.splitext(input_file)[0] + '_roads.tif')
        save_tif(
            filename=output_filename,
            data_arrays=[roads],
            band_names=['ROADS'],
            data_type=gdal.GDT_Float32,
            geotransform=geotransform,
            projection=projection
        )

    dataset = None

