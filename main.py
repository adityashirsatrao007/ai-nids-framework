import os
import pandas as pd
import numpy as np
from src.data.downloader import main as download_dataset
from src.preprocessing.cleaner import clean_data, basic_preprocessing
from src.feature_selection.selector import select_features_mutual_info, plot_feature_importance
from src.models.hybrid_model import get_ensemble_model, CNNLSTMModel
from src.training.trainer import train_pytorch_model
from src.evaluation.metrics import calculate_metrics, print_report, export_results
from src.visualization.plotter import plot_confusion_matrix, plot_roc_curve, plot_training_history
import torch
import joblib

def run_pipeline():
    # 1. Setup & Download
    print("--- Starting NIDS Research Pipeline ---")
    if not os.path.exists("dataset/MachineLearningCSV"):
        download_dataset()
    
    # 2. Data Loading
    print("Loading Real Large-Scale Dataset (2.8M rows)...")
    data_path = "dataset/MachineLearningCSV/real_data_hf.csv"
    if not os.path.exists(data_path):
        print("Real data not found. Running acquisition script...")
        from src.data.hf_downloader import download_real_dataset
        data_path = download_real_dataset()
    
    # Use caching for sampled dataset to speed up re-runs
    cache_path = "results/ready_sample.csv"
    if os.path.exists(cache_path):
        print(f"Loading cached 500k sample from {cache_path}...")
        df = pd.read_csv(cache_path)
    else:
        # Use chunking or memory optimization for large file
        # Stratified Sampling: 500k target to fit in memory
        print("Reading and Sampling massive dataset...")
        df = pd.read_csv(data_path, low_memory=False)
        df = clean_data(df)
        
        sample_size = 500000
        if len(df) > sample_size:
            print(f"Sampling {sample_size} rows with attack priority...")
            # Use an explicit loop to ensure the 'Label' column is never lost in grouping
            sampled_groups = []
            for label, group in df.groupby('Label'):
                target_n = int(sample_size * (len(group) / len(df)))
                # Ensure at least 2000 samples per class if possible
                n_to_sample = min(len(group), max(2000, target_n))
                sampled_groups.append(group.sample(n=n_to_sample, random_state=42))
            
            df = pd.concat(sampled_groups).reset_index(drop=True)
            
            # Clip to total sample size if it grew too much, while keeping distribution
            if len(df) > sample_size * 1.5:
                 df = df.sample(n=sample_size, random_state=42)
        
        # Final safety check
        if 'Label' not in df.columns:
            raise ValueError("Critical Error: 'Label' column lost during sampling. Check CSV headers.")
            
        print(f"Saving high-fidelity sampled dataset to {cache_path}...")
        df.to_csv(cache_path, index=False)

    
    print(f"Dataset ready. Shape: {df.shape}")
    X, y, label_encoder = basic_preprocessing(df)
    
    # 4. Feature Selection (With Caching)
    selected_features_path = 'results/selected_features.joblib'
    if os.path.exists(selected_features_path):
        print(f"Loading cached top 20 features from {selected_features_path}...")
        top_features = joblib.load(selected_features_path)
    else:
        print("Calculating Mutual Information for 78 features...")
        top_features, importances = select_features_mutual_info(X, y, top_n=20)
        joblib.dump(top_features, selected_features_path)
    
    X_selected = X[top_features]
    
    # 5. Train-Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Ensemble Model (RF + XGB)
    print("\n--- Training Ensemble Model (RF + XGB) ---")
    ensemble = get_ensemble_model()
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    
    # Robust probability extraction for Multi-Class ROC curves
    try:
        y_prob_ens = ensemble.predict_proba(X_test)
    except:
        print("Warning: Ensemble probability extraction failed. Using XGB component.")
        y_prob_ens = ensemble.named_estimators_['xgb'].predict_proba(X_test)
    
    # Evaluate with full probability matrix for OvR ROC-AUC
    metrics_ens = calculate_metrics(y_test, y_pred_ens, y_prob_ens)
    print("Ensemble Metrics:", metrics_ens)
    export_results(metrics_ens, "results/ensemble_metrics.csv")

    
    # 7. CNN-LSTM Model
    print("\n--- Training Deep Learning Model (CNN-LSTM) ---")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'results/scaler.joblib')
    
    num_classes = len(np.unique(y))
    dl_model = CNNLSTMModel(input_dim=len(top_features), num_classes=num_classes)
    # Optimized epochs for 500k records
    history = train_pytorch_model(dl_model, X_train_scaled, y_train, X_test_scaled, y_test, epochs=5, batch_size=128)
    
    # 8. Per-Class Metrics Reporting
    print("\n--- Detailed Per-Attack Analysis ---")
    from sklearn.metrics import classification_report
    # Inverse transform labels to their original names for the final paper
    y_test_names = label_encoder.inverse_transform(y_test)
    y_pred_names = label_encoder.inverse_transform(y_pred_ens)
    
    # Sanitize names to prevent Windows console Unicode errors with CICIDS2017 characters
    y_test_names = [str(n).encode('ascii', 'replace').decode('ascii') for n in y_test_names]
    y_pred_names = [str(n).encode('ascii', 'replace').decode('ascii') for n in y_pred_names]
    
    report = classification_report(y_test_names, y_pred_names)
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    
    # Save models
    torch.save(dl_model.state_dict(), 'results/cnn_lstm_v1.pth')
    joblib.dump(ensemble, 'results/ensemble_v1.joblib')
    
    # 9. SHAP Explainer (Save for UI)
    print("\n--- Initializing Stable SHAP Explainer ---")
    import shap
    
    # Use the already trained XGBoost model from the ensemble
    it_model = ensemble.named_estimators_['xgb']
    explainer = shap.TreeExplainer(it_model)
    joblib.dump(explainer, 'results/explainer.joblib')
    joblib.dump(top_features, 'results/feature_names.joblib')
    
    # 10. Visualization
    safe_classes = [str(c).encode('ascii', 'replace').decode('ascii') for c in label_encoder.classes_]
    plot_confusion_matrix(y_test, y_pred_ens, safe_classes, "results/cm_ensemble.png")
    plot_training_history(history, "results/training_history.png")
    
    print("\n--- Pipeline Complete. Results saved in 'results/' ---")

if __name__ == "__main__":
    run_pipeline()
