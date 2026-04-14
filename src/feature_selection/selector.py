import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def select_features_mutual_info(X, y, top_n=20):
    """
    Selects top N features using Mutual Information.
    """
    print(f"Calculating Mutual Information for {X.shape[1]} features...")
    importances = mutual_info_classif(X, y)
    feat_importances = pd.Series(importances, index=X.columns)
    
    top_features = feat_importances.nlargest(top_n).index.tolist()
    print(f"Top {top_n} features selected: {top_features}")
    
    return top_features, feat_importances

def select_features_rfe(X, y, top_n=20):
    """
    Selects top N features using Recursive Feature Elimination with Random Forest.
    """
    print(f"Running RFE for top {top_n} features (this may take a while)...")
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=top_n, step=5)
    selector = selector.fit(X, y)
    
    top_features = X.columns[selector.support_].tolist()
    return top_features

def plot_feature_importance(feat_importances, top_n=20, save_path=None):
    """
    Plots the top N features by importance.
    """
    plt.figure(figsize=(10, 8))
    feat_importances.nlargest(top_n).plot(kind='barh')
    plt.title(f"Top {top_n} Features (Mutual Information)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
