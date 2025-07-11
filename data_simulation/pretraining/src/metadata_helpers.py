import pyarrow.parquet as pq
import pandas as pd
import geopandas as gpd
from pathlib import Path
import urllib.request
import fsspec
from fsspec.parquet import open_parquet_file
from io import BytesIO
from PIL import Image
from rasterio.io import MemoryFile
# from tqdm.notebook import tqdm
from tqdm import tqdm
import os

from aiohttp.client_exceptions import ClientResponseError

from .sample_helpers import *

def metadata_from_url(access_url, local_url):
    local_url, response = urllib.request.urlretrieve(access_url, local_url)
    df = pq.read_table(local_url).to_pandas()
    df['timestamp'] = pd.to_datetime(df.timestamp)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.centre_lon, df.centre_lat), crs=df.crs.iloc[0]
    )
    return gdf

def filter_metadata(df,
                    region=None,
                    daterange=None,
                    cloud_cover=(0,100),
                    nodata=(0, 1.0)
                   ):
    """Filters the Major-TOM dataframe based on several parameters

    Args:
        df (geopandas dataframe): Parent dataframe
        region (shapely geometry object) : Region of interest
        daterange (tuple) : Inclusive range of dates (example format: '2020-01-01')
        cloud_cover (tuple) : Inclusive percentage range (0-100) of cloud cover
        nodata (tuple) : Inclusive fraction (0.0-1.0) of no data allowed in a sample

    Returns:
        df: a filtered dataframe
    """
    # temporal filtering
    if daterange is not None:
        assert (isinstance(daterange, list) or isinstance(daterange, tuple)) and len(daterange)==2
        df = df[df.timestamp >= daterange[0]]
        df = df[df.timestamp <= daterange[1]]
    
    # spatial filtering
    if region is not None:
        idxs = df.sindex.query(region)
        df = df.take(idxs)
    # cloud filtering
    if cloud_cover is not None:
        df = df[df.cloud_cover >= cloud_cover[0]]
        df = df[df.cloud_cover <= cloud_cover[1]]

    # spatial filtering
    if nodata is not None:
        df = df[df.nodata >= nodata[0]]
        df = df[df.nodata <= nodata[1]]

    return df

def read_row(row, columns=["thumbnail"]):
    """Reads a row from a Major-TOM dataframe

    Args:
        row (row from geopandas dataframe): The row of metadata
        columns (list): columns to be read from the file

    Returns:
        data (dict): dictionary with returned data from requested columns
    """
    with open_parquet_file(row.parquet_url,columns = columns) as f:
        with pq.ParquetFile(f) as pf:
            row_group = pf.read_row_group(row.parquet_row, columns=columns)

    if columns == ["thumbnail"]:
        stream = BytesIO(row_group['thumbnail'][0].as_py())
        return Image.open(stream)
    else:
        row_output = {}
        for col in columns:
            bytes = row_group[col][0].as_py()

            if col != 'thumbnail':
                row_output[col] = read_tif_bytes(bytes)
            else:
                stream = BytesIO(bytes)
                row_output[col] = Image.open(stream)

        return row_output

def filter_download(df, local_dir, source_name, by_row=False, verbose=False, tif_columns=None):
    """Downloads and unpacks the data of Major-TOM based on a metadata dataframe

    Args:
        df (geopandas dataframe): Metadata dataframe
        local_dir (str or Path): Path to where the data is to be stored locally
        source_name (str): Name alias of the resulting dataset
        by_row (bool): If True, it will access individual rows of parquet via HTTP; otherwise, entire parquets are downloaded temporarily
        verbose (bool): Option for potential internal state printing
        tif_columns (list of str): Optionally specified columns to be downloaded as .tifs, e.g., ['B04', 'B03', 'B02']

    Returns:
        None
    """

    if isinstance(local_dir, str):
        local_dir = Path(local_dir)

    temp_file = local_dir / 'temp.parquet'

    # Identify all parquets that need to be downloaded (group them)
    urls = df.parquet_url.unique()
    if verbose:
        print(f'Starting download of {len(urls)} parquet files.')

    for url in tqdm(urls, desc='Downloading and unpacking...'):
        # Identify all relevant rows
        rows = df[df.parquet_url == url].parquet_row.unique()

        try:
            if not by_row:  # (downloads entire parquet)
                # Download a temporary file
                temp_path, _ = urllib.request.urlretrieve(url, temp_file)
            else:
                f = fsspec.open(url)
                temp_path = f.open()
        except Exception as e:
            if verbose:
                print(f"Failed to download parquet from {url}. Error: {e}. Skipping.")
            continue  # Skip to the next URL

        try:
            with pq.ParquetFile(temp_path) as pf:
                for row_idx in rows:
                    try:
                        table = pf.read_row_group(row_idx)

                        product_id = table['product_id'][0].as_py()
                        grid_cell = table['grid_cell'][0].as_py()
                        row = grid_cell.split('_')[0]

                        dest = local_dir / Path(f"{source_name}/{row}/{grid_cell}/{product_id}")
                        dest.mkdir(exist_ok=True, parents=True)

                        # Determine which columns to process
                        if tif_columns is None:
                            columns = [col for col in table.column_names if col.startswith('B')] + ['cloud_mask']
                        else:
                            columns = tif_columns

                        # Save .tif files
                        for col in columns:
                            with open(dest / f"{col}.tif", "wb") as f_out:
                                f_out.write(table[col][0].as_py())

                        # Save thumbnail as .png
                        thumbnail_col = 'thumbnail'
                        if thumbnail_col in table.column_names:
                            with open(dest / f"{thumbnail_col}.png", "wb") as f_out:
                                f_out.write(table[thumbnail_col][0].as_py())
                        else:
                            if verbose:
                                print(f"Thumbnail column missing for row {row_idx} in {url}.")
                    
                    except ClientResponseError as cre:
                        if verbose:
                            print(f"ClientResponseError for row {row_idx} in {url}: {cre}. Skipping this row.")
                        continue  # Skip this row and continue with the next
                    except Exception as e:
                        if verbose:
                            print(f"Error processing row {row_idx} in {url}: {e}. Skipping this row.")
                        continue  # Skip this row and continue with the next

        except ClientResponseError as cre:
            if verbose:
                print(f"ClientResponseError while reading parquet file from {url}: {cre}. Skipping this parquet.")
            continue  # Skip this parquet file
        except Exception as e:
            if verbose:
                print(f"Error reading parquet file from {url}: {e}. Skipping this parquet.")
            continue  # Skip this parquet file
        finally:
            if not by_row:
                # Remove downloaded temporary file
                try:
                    os.remove(temp_path)
                except OSError as oe:
                    if verbose:
                        print(f"Error removing temporary file {temp_path}: {oe}")
            else:
                try:
                    f.close()
                except Exception as e:
                    if verbose:
                        print(f"Error closing file handle for {url}: {e}")
