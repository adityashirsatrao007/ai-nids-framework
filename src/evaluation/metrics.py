from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates standard research metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
    }
    
    if y_prob is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics['ROC-AUC'] = 0.0
            
    return metrics

def export_results(metrics, filename):
    """
    Exports metrics to a CSV file.
    """
    df = pd.DataFrame([metrics])
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")

def print_report(y_true, y_pred, label_encoder=None):
    """
    Prints classification report and confusion matrix with original attack names.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    if label_encoder is not None:
        # Convert numeric labels back to original attack names
        y_true_names = label_encoder.inverse_transform(y_true)
        y_pred_names = label_encoder.inverse_transform(y_pred)
        print("\nClassification Report (Per-Attack Analysis):")
        print(classification_report(y_true_names, y_pred_names))
    else:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


