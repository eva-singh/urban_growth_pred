"""
Model Evaluation Script - Calculates comprehensive accuracy metrics
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
import seaborn as sns
import _0_config as cfg


def load_test_data():
    """Load test data"""
    test_X = np.load(os.path.join(cfg.OUT_NP_DIR, 'test_X.npy'))
    test_Y = np.load(os.path.join(cfg.OUT_NP_DIR, 'test_Y.npy'))
    
    test_X = test_X[..., None]  
    test_Y = np.squeeze(test_Y)  
    
    return test_X, test_Y


def calculate_metrics(y_true, y_pred):
    """Calculate various accuracy metrics"""
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # intersection over union
    iou = jaccard_score(y_true_flat, y_pred_flat, zero_division=0)
    
    
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Urban', 'Urban'],
                yticklabels=['Non-Urban', 'Urban'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_predictions(test_X, test_Y, predictions, n_samples=4, save_path=None):
    """Plot sample predictions with metrics"""
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    for i in range(min(n_samples, len(test_X))):
        
        sample_acc = accuracy_score(test_Y[i].flatten(), predictions[i].flatten())
        sample_iou = jaccard_score(test_Y[i].flatten(), predictions[i].flatten(), zero_division=0)
        
        
        axes[i, 0].imshow(test_X[i, -1, ..., 0], cmap='gray')
        axes[i, 0].set_title('Last Input Year')
        axes[i, 0].axis('off')
        
        
        axes[i, 1].imshow(test_Y[i], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        
        axes[i, 2].imshow(predictions[i], cmap='gray')
        axes[i, 2].set_title(f'Prediction\nAcc: {sample_acc:.3f}')
        axes[i, 2].axis('off')
        
       
        diff = predictions[i].astype(int) - test_Y[i].astype(int)
        axes[i, 3].imshow(diff, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[i, 3].set_title(f'Difference\nIoU: {sample_iou:.3f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model_path=None, visualize=True):
    """Main evaluation function"""
   
    if model_path is None:
        
        possible_paths = [
            os.path.join(cfg.OUT_DIR, 'best_model.keras'),
            os.path.join(cfg.OUT_DIR, 'best_model.h5'),
            os.path.join(cfg.OUT_DIR, 'final_model.keras'),
            os.path.join(cfg.OUT_DIR, 'final_model.h5'),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"No model found in {cfg.OUT_DIR}. "
                f"Please train the model first by running 3_train.py"
            )
    
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, compile=False)
    
    
    print("Loading test data...")
    test_X, test_Y = load_test_data()
    print(f"Test data shape: X={test_X.shape}, Y={test_Y.shape}")
    
    
    print("\nMaking predictions...")
    predictions_prob = model.predict(test_X, batch_size=4, verbose=1)
    predictions = (predictions_prob[..., 0] > 0.5).astype(int)
    
    
    print("\nCalculating metrics...")
    metrics = calculate_metrics(test_Y, predictions)
    
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print("="*50)
    
    
    metrics_file = os.path.join(cfg.OUT_DIR, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"IoU:       {metrics['iou']:.4f}\n")
        f.write("="*50 + "\n")
    
    print(f"\nMetrics saved to: {metrics_file}")
    
    
    if visualize:
        print("\nGenerating visualizations...")
        
        #cpnf
        cm_path = os.path.join(cfg.OUT_DIR, 'confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
        
        #sample
        samples_path = os.path.join(cfg.OUT_DIR, 'sample_predictions.png')
        plot_sample_predictions(test_X, test_Y, predictions, n_samples=4, save_path=samples_path)
        print(f"Sample predictions saved to: {samples_path}")
    
    return metrics, predictions


if __name__ == '__main__':
    metrics, predictions = evaluate_model(visualize=True)