"""Data preparation script:
- Lists tif files, crops to AOI, resizes to IMG_SHAPE
- Builds temporal windows: IN_SEQ -> predict next year
- Saves train/val/test numpy arrays (X, Y) as .npy files
"""
import os, numpy as np, random, re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import list_tif_files, crop_to_bbox, read_tif_as_array, resize_image, normalize_binary, save_npy
import _0_config as cfg


def prepare_sequences(file_paths, bbox=None, img_shape=(256,256), in_seq=5, pred_steps=1):
    """Return arrays X and Y:
    X shape: (N_windows, in_seq, H, W)
    Y shape: (N_windows, pred_steps, H, W)
    """

    valid_files = []
    valid_years = []

   
    for p in file_paths:
        fname = os.path.basename(p)
        digits = ''.join(ch for ch in fname if ch.isdigit())

        #4 digit year
        year = None
        for i in range(len(digits)-3):
            chunk = digits[i:i+4]
            if 1900 <= int(chunk) <= 2100:
                year = int(chunk)
                break

        if year is None:
            #skip files corr "sQ_urbanMap_global_stackTS.tif"
            print(f"Skipping non-year file: {fname}")
            continue

        valid_files.append(p)
        valid_years.append(year)

  
    valid_files = np.array(valid_files)
    valid_years = np.array(valid_years)

    idx = np.argsort(valid_years)
    valid_files = valid_files[idx]
    valid_years = valid_years[idx]

    arrays = []

    for p in tqdm(valid_files, desc='Reading TIFFs'):
        if bbox is not None:
            arr = crop_to_bbox(p, bbox)
        else:
            arr = read_tif_as_array(p)

        arr = resize_image(arr, img_shape)
        arr = normalize_binary(arr)
        arrays.append(arr)

    arrays = np.stack(arrays, axis=0)  # (T, H, W)

    T = arrays.shape[0]
    X_list, Y_list = [], []

    for i in range(T - in_seq - pred_steps + 1):
        X_list.append(arrays[i:i+in_seq])
        Y_list.append(arrays[i+in_seq:i+in_seq+pred_steps])

    X = np.stack(X_list)
    Y = np.stack(Y_list)

    sample_years = valid_years[in_seq : in_seq + X.shape[0]]

    return X, Y, valid_years, sample_years




def main():
    files = list_tif_files(cfg.DATA_DIR)
    if len(files) == 0:
        raise ValueError(f"No tif files found in {cfg.DATA_DIR}. Put the yearly .tif files there.")


    X, Y, years, sample_years = prepare_sequences(
        files,
        bbox=cfg.AOI_BBOX,
        img_shape=cfg.IMG_SHAPE,
        in_seq=cfg.IN_SEQ,
        pred_steps=cfg.PRED_STEPS
    )

   
    N = X.shape[0]
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)

    train_X, train_Y = X[:n_train], Y[:n_train]
    val_X, val_Y = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    test_X, test_Y = X[n_train+n_val:], Y[n_train+n_val:]

    #split
    train_years = sample_years[:n_train]
    val_years   = sample_years[n_train:n_train+n_val]
    test_years  = sample_years[n_train+n_val:]

  
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'train_X.npy'), train_X)
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'train_Y.npy'), train_Y)
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'val_X.npy'), val_X)
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'val_Y.npy'), val_Y)
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'test_X.npy'), test_X)
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'test_Y.npy'), test_Y)

    #new
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'all_years.npy'), years)
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'train_years.npy'), train_years)
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'val_years.npy'), val_years)
    save_npy(os.path.join(cfg.OUT_NP_DIR, 'test_years.npy'), test_years)

    print('Saved Numpy arrays to', cfg.OUT_NP_DIR)



if __name__ == '__main__':
    main()
