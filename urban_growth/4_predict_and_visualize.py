"""
Option C – Predict ANY future year + visualize historical test results.

This script now supports:
1. Normal test-set visualization (your original function).
2. Future prediction for any user-provided target year.
3. Recursive year-by-year forecasting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import _0_config as cfg



def load_test():
    tx = np.load(os.path.join(cfg.OUT_NP_DIR, 'test_X.npy'))
    ty = np.load(os.path.join(cfg.OUT_NP_DIR, 'test_Y.npy'))

    tx = tx[..., None]       # (N, seq, H, W, 1)
    ty = ty[..., 0]          # (N, H, W)

    return tx, ty


def load_years():
    years = np.load(os.path.join(cfg.OUT_NP_DIR, 'all_years.npy'))
    return years



def vis_sample(model_path=None, n=4):
    tx, ty = load_test()

    if model_path is None:
        model_path = os.path.join(cfg.OUT_DIR, 'best_model.h5')

    model = load_model(model_path, compile=False)
    preds = model.predict(tx, batch_size=4)
    preds = (preds[..., 0] > 0.5).astype('int')

    for i in range(min(n, tx.shape[0])):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].imshow(tx[i, -1, ..., 0], cmap='gray')
        axs[0].set_title('Last input year')

        axs[1].imshow(ty[i], cmap='gray')
        axs[1].set_title('True next year')

        axs[2].imshow(preds[i], cmap='gray')
        axs[2].set_title('Predicted next year')

        axs[3].imshow(preds[i] - ty[i], cmap='bwr')
        axs[3].set_title('Pred - True')

        for a in axs:
            a.axis('off')
        plt.show()



def predict_future(target_year):
    """
    Predict future frames recursively until reaching target_year.
    """

    # Load trained model
    model_path = os.path.join(cfg.OUT_DIR, "best_model.h5")
    model = load_model(model_path, compile=False)

    
    years = load_years()
    last_available_year = years[-1] 

    print(f"\nLast available real year in dataset: {last_available_year}")

    if target_year <= last_available_year:
        print("\nTarget year is NOT in the future. Nothing to predict.")
        return

    
    test_X = np.load(os.path.join(cfg.OUT_NP_DIR, 'test_X.npy'))
    last_seq = test_X[-1]#shape(seq, H, W)
    last_seq = last_seq[..., None]  

    current_seq = last_seq.copy()
    current_year = last_available_year

    future_predictions = []
    future_years = []

   
    while current_year < target_year:
        
        pred = model.predict(current_seq[np.newaxis, ...])
        pred = (pred[0, ..., 0] > 0.5).astype("int")

        future_predictions.append(pred)
        future_years.append(current_year + 1)

        current_year += 1

        
        pred_with_channel = pred[..., None] 
        
        
        
        pred_for_concat = pred_with_channel[np.newaxis, ...] 
        
        
        current_seq = np.concatenate(
            [current_seq[1:], pred_for_concat],
            axis=0
        )

    #save pred
    future_predictions = np.array(future_predictions)
    future_years = np.array(future_years)

    print("\nPrediction completed.")
    print("Predicted years:", future_years)

    return future_years, future_predictions


def plot_future_predictions(future_years, preds):
    """
    Plots only the final predicted year.
    To plot all years, change the index from -1 to iterate over len(future_years).
    """
    
    if len(future_years) == 0:
        print("No future predictions were generated.")
        return

    
    last_index = -1
    last_year = future_years[last_index]
    last_pred = preds[last_index]

    plt.figure(figsize=(6, 6)) 
    plt.imshow(last_pred, cmap='gray')
    plt.title(f"Predicted Urban Map – {last_year}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":

    print("\n===== URBAN GROWTH PREDICTION TOOL =====\n")
    print("1. Visualize test set predictions")
    print("2. Predict FUTURE years")
    choice = input("\nEnter choice (1 or 2): ").strip()

   
    if choice == "1":
        vis_sample()

    
    elif choice == "2":
        target_year = int(input("Enter future target year (e.g., 2035): ").strip())

        fy, preds = predict_future(target_year)
        plot_future_predictions(fy, preds)

    else:
        print("Invalid choice.")
