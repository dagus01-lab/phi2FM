import py7zr
import os

# Path to the first file
first_part = "/Data/phisatnet_clouds/results.zarr.zip.001"

# Output directory
output_dir = "clouds.zarr"

# Extract the archive
with py7zr.SevenZipFile(first_part, mode='r') as archive:
    archive.extractall(path=output_dir)

print(f"Extraction completed to folder: {output_dir}")
