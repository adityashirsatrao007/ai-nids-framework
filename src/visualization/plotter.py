import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """
    Plots a research-grade confusion matrix heatmap.
    'labels' should be the original string names from label_encoder.classes_
    """
    import numpy as np
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Map back to string names for the axis (using the indices found in unique_labels)
    display_labels = [str(labels[i]) for i in unique_labels]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=display_labels, yticklabels=display_labels)
    plt.title('Confusion Matrix Heatmap (Multi-Attack Analysis)')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plots the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_training_history(history, save_path=None):
    """
    Plots loss and accuracy curves for deep learning models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_ablation_results(results_df, save_path=None):
    """
    Plots a comparative bar chart for the ablation study.
    results_df should have 'Model' and 'Accuracy' columns.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    
    # Sort for visual progression
    results_df = results_df.sort_values(by='Accuracy')
    
    ax = sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
    
    # Add accuracy labels on top of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
                    
    plt.title('Ablation Study: Standalone vs Hybrid Architecture Accuracy')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Architecture')
    plt.ylim(min(results_df['Accuracy']) - 0.05, 1.05) # Add padding
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()    
