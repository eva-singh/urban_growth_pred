# Satellite Image Analysis for Urban Growth Prediction - Project Code Bundle

This bundle contains step-by-step code to process the annual global `.tif` urban extent dataset, prepare training data, build a ConvLSTM model, train it, evaluate it, and visualize predictions.

## Structure
- `requirements.txt` - Python packages to install
- `_0_config.py` - global configuration (paths, AOI)
- `1_data_prep.py` - load, crop, resample, create sequences, save `.npy` tensors
- `_2_model.py` - Keras ConvLSTM model definition
- `3_train.py` - training loop, callbacks, evaluation metrics
- `4_predict_and_visualize.py` - inference and visualization helpers
- `5_evaluate_model.py` - evaluation metrics
- `app.py` - for a simple dashboard
- `utils.py` - utility functions used across scripts
- `run_example.sh` - example commands to run the pipeline
- `README.md` - this file
