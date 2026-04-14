import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hybrid_model import get_ensemble_model, CNNLSTMModel
from training.trainer import train_pytorch_model
from visualization.plotter import plot_ablation_results
from preprocessing.cleaner import basic_preprocessing

def run_ablation_study():
    print("--- Starting Hybrid Architecture Ablation Study ---")
    
    # 1. Load the pre-processed high-fidelity dataset
    sample_path = 'results/ready_sample.csv'
    features_path = 'results/selected_features.joblib'
    
    if not os.path.exists(sample_path) or not os.path.exists(features_path):
        print("Missing dataset caches. Please run main.py first.")
        return
        
    print("Loading high-fidelity dataset and features...")
    df = pd.read_csv(sample_path)
    X, y, label_encoder = basic_preprocessing(df)
    
    top_features = joblib.load(features_path)
    X_selected = X[top_features]
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # Set up neural network scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    num_classes = len(np.unique(y))
    
    # ==========================
    # Study 1: Random Forest Only
    # ==========================
    print("\nTraining Standalone Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    results.append({'Model': 'Random Forest', 'Accuracy': rf_acc})
    
    # ==========================
    # Study 2: XGBoost Only
    # ==========================
    print("Training Standalone XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, tree_method='hist', n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    results.append({'Model': 'XGBoost', 'Accuracy': xgb_acc})
    
    # ==========================
    # Study 3: Proposed Hybrid Ensemble (RF + XGB)
    # ==========================
    print("Training Our Proposed Hybrid Ensemble...")
    ensemble = get_ensemble_model()
    ensemble.fit(X_train, y_train)
    ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
    results.append({'Model': 'Hybrid Ensemble (Ours)', 'Accuracy': ens_acc})
    
    # ==========================
    # Study 4: CNN-LSTM
    # ==========================
    print("Training CNN-LSTM...")
    dl_model = CNNLSTMModel(input_dim=len(top_features), num_classes=num_classes)
    # Reduced epochs for rapid ablation testing
    history = train_pytorch_model(dl_model, X_train_scaled, y_train, X_test_scaled, y_test, epochs=3, batch_size=128)
    cnn_acc = history['val_acc'][-1]
    
    results.append({'Model': 'CNN-LSTM Temporal Base', 'Accuracy': cnn_acc})
    
    # ==========================
    # Finalize & Plot
    # ==========================
    results_df = pd.DataFrame(results)
    print("\n--- Ablation Study Results ---")
    print(results_df)
    
    results_df.to_csv('results/ablation_results.csv', index=False)
    print("Results saved to results/ablation_results.csv")
    
    # Plot without showing UI
    import matplotlib
    matplotlib.use('Agg') 
    plot_ablation_results(results_df, save_path='results/ablation_comparison.png')
    print("Chart saved to results/ablation_comparison.png")

if __name__ == "__main__":
    run_ablation_study()
