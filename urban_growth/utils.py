import os, glob, numpy as np, rasterio
from rasterio.windows import from_bounds
from skimage.transform import resize
from tqdm import tqdm

def list_tif_files(data_dir):
    # Assumes filenames contain the year number; sorts by name
    files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    return files

def read_tif_as_array(path, band=1):
    # Returns a 2D numpy array of the requested band
    with rasterio.open(path) as src:
        arr = src.read(band).astype('float32')
        # If there is a nodata value, convert to 0
        if src.nodata is not None:
            arr[arr == src.nodata] = 0
    return arr

def crop_to_bbox(path, bbox):
    # bbox in (lon_min, lat_min, lon_max, lat_max)
    with rasterio.open(path) as src:
        window = from_bounds(*bbox, src.transform)
        arr = src.read(1, window=window).astype('float32')
    return arr

def resize_image(img, out_shape):
    # input: 2D array
    return resize(img, out_shape, order=0, preserve_range=True, anti_aliasing=False).astype('float32')

def normalize_binary(img):
    # Ensure binary 0 or 1 values (threshold at 0.5)
    im = img.copy()
    im = (im > 0.5).astype('float32')
    return im

def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def load_npy(path):
    return np.load(path)
