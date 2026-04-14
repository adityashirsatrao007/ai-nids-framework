import shap
import matplotlib.pyplot as plt
import numpy as np
import torch

def explain_tree_model(model, X_sample, feature_names, save_path=None):
    """
    Explains tree-based models (RF, XGB) using SHAP TreeExplainer.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(f"{save_path}_summary.png")
    plt.show()
    
    return explainer, shap_values

def explain_pytorch_model(model, X_sample, feature_names, save_path=None):
    """
    Explains PyTorch model using SHAP DeepExplainer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X_sample.values, dtype=torch.float32).to(device)
    
    # We use a background dataset for DeepExplainer (using a small subset)
    background = X_tensor[:100]
    explainer = shap.DeepExplainer(model, background)
    
    shap_values = explainer.shap_values(X_tensor)
    
    plt.figure()
    # DeepExplainer output for multiclass is a list
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(f"{save_path}_summary_dl.png")
    plt.show()
    
    return explainer, shap_values

def local_explanation(explainer, shap_values, instance_idx, feature_names, save_path=None):
    """
    Generates a waterfall plot for a single prediction.
    """
    plt.figure()
    # Note: SHAP API varies, using force_plot for compatibility if waterfall fails
    shap.plots.force(explainer.expected_value[1], shap_values[1][instance_idx], feature_names=feature_names, matplotlib=True)
    if save_path:
        plt.savefig(f"{save_path}_local_{instance_idx}.png")
    plt.show()
