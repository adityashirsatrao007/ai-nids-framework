# Explainable Hybrid Machine Learning for Network Intrusion Detection (NIDS)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![IEEE](https://img.shields.io/badge/Manuscript-IEEE-blue)](paper/IEEE_Paper_Draft.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Solving the Black-Box Problem in Cybersecurity"** - A dual-stream CNN-LSTM and Ensemble framework with SHAP-based local feature attribution, achieving 99.09% accuracy on the CICIDS2017 dataset.

---

## 📌 Overview
Modern Network Intrusion Detection Systems (NIDS) face two major hurdles: the **"Opacity Crisis"** (black-box deep learning) and **Severe Class Imbalance**. This repository contains the official implementation of a hybrid framework designed to solve both. 

By combining **spatial-temporal rhythms** (via CNN-LSTM) with **structural ensembles** (via XGBoost/Random Forest), we achieve near-perfect detection rates. Crucially, we utilize **SHAP (Shapley Additive Explanations)** to provide local, per-packet transparency, allowing security analysts to verify exactly *why* a threat was flagged.

### 🏗️ Architecture
The system employs a dual-stream engine processing 20 high-fidelity features selected through Mutual Information (MI).

![Workflow](figures/fig2_workflow_diagram_final.png)
*Fig 1: Architecture of the Explainable Hybrid NIDS Workflow.*

---

## 🚀 Key Features
- **Dual-Stream Hybrid Engine**: Parallel processing of temporal rhythms (DL) and structural signatures (Ensemble).
- **SHAP-Based Explainability**: Native integration of `TreeExplainer` for local feature attribution and forensic waterfall plots.
- **Attack-Priority Stratification**: A custom sampling approach to ensure rare attacks (like Heartbleed and Infiltration) are never ignored during training.
- **MI-RFE Selection**: Reduced feature space from 78 to 20 dimensions for optimized real-time performance.

---

## 📊 Performance
The framework was validated against the **CICIDS2017** dataset, achieving a 99.09% overall accuracy.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99.09% |
| **Weighted Precision** | 99.00% |
| **Weighted Recall** | 99.08% |
| **Weighted F1-Score** | 99.04% |

![ROC Curve](figures/fig3_roc_auc_curve.png)
*Fig 2: Receiver Operating Characteristic (AUC = 0.99).*

---

## 🛠️ Usage

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Data Acquisition
The pipeline automatically downloads and samples the CICIDS2017 dataset from official sources. Configure your `dataset/` path if using a local copy.

### 3. Run the Research Pipeline
Execute the full end-to-end experiment (Preprocessing -> Selection -> Training -> Evaluation -> SHAP Export):
```bash
python main.py
```

### 4. Interactive Demo
A Dockerized UI for real-time traffic simulation and SHAP visualization is included in the `demo/` folder.
```bash
cd demo
docker build -t nids-demo .
docker run -p 8501:8501 nids-demo
```

---

## 📖 Citation
If you use this work in your research, please cite our IEEE paper:

```bibtex
@inproceedings{shirsatrao2025explainable,
  title={Explainable Hybrid Machine Learning Framework for Network Intrusion Detection Using SHAP Analysis},
  author={Shirsatrao, Aditya Vishal and Sangolgi, Vijay A and Patil, M.B.},
  booktitle={2025 7th International Conference on Signal Processing, Computing and Control (ISPCC)},
  year={2025}
}
```

---

## 🤝 Contributing
This is an open research project. Contributions to the `src/` modules or additional dataset benchmarks (e.g., UNSW-NB15) are welcome. 

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
