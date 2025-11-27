#DATA_DIR to point where your .tif files are stored
import os


DATA_DIR = os.path.expanduser(r"C:\Users\lenovo\Downloads\16602224 (1)")  # <-- change me

#directory to save cropped region tifs
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
OUT_NP_DIR = os.path.join(OUT_DIR, "npy")
os.makedirs(OUT_NP_DIR, exist_ok=True)

#area of interest (AOI) bounding box
#delhi bounding box
AOI_BBOX = (76.90, 28.40, 77.35, 28.90)


IMG_SHAPE = (256, 256)

#number of past years to use as input, and how many years to predict
IN_SEQ = 5
PRED_STEPS = 1


RANDOM_SEED = 42
