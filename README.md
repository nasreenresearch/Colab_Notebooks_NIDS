# 🧠 Colab_Notebooks_NIDS  
**AI-Driven Network Intrusion Detection, XAI, and Adversarial Security Framework**

This repository provides a unified set of **Google Colab notebooks** for research and experimentation on **Network Intrusion Detection Systems (NIDS)**.  
It integrates **malware and intrusion detection datasets**, **deep learning architectures**, **Explainable AI (XAI)**, and **Adversarial AI** modules to advance intelligent, interpretable, and resilient cybersecurity analytics.

---

## 🔍 Overview
`Colab_Notebooks_NIDS` supports full-stack IDS experimentation, from data ingestion and preprocessing to model training, adversarial testing, and explainability visualization.  
It was developed as part of the research thesis *“Resilient AI-Driven Network Intrusion Detection System for Emerging Threats”* (Deakin University & VIT Chennai, 2025).

---

## 🧩 Datasets Included
The notebooks provide loaders, preprocessing pipelines, and benchmark splits for multiple open datasets:

| Dataset | Description |
|----------|--------------|
| **NSL-KDD** | Classic benchmark for connection-based intrusion detection |
| **CIC-IDS-2017 / 2018** | Realistic traffic with diverse DoS/DDoS and Brute-Force attacks |
| **Bot-IoT-2018** | IoT-focused dataset for network and device exploitation |
| **UNSW-NB15** | Hybrid attack dataset for modern network environments |
| **IoT-23** | Malware traffic from IoT devices |
| **Custom Malware Samples** | Preprocessed malicious traces for behavioral learning |

---

## 🧠 Model Architectures
- **Deep Neural Networks (DNN)**
- **Convolutional Networks (CNN)**
- **Gated Recurrent Units (GRU)**
- **Variational Autoencoders (VAE)**
- **Hybrid & Continual Learning Pipelines (CL, Replay, EWC, GEM)**
- **Adversarial AI Frameworks** (FGSM, PGD, CW attacks)

---

## ⚙️ Features
- Automated preprocessing and feature normalization for all datasets  
- Task-based incremental learning and continual training workflows  
- Explainable AI visualization via **LIME**, **SHAP**, and **Grad-CAM**  
- Adversarial robustness testing and defense evaluation  
- Log correlation and anomaly scoring utilities  
- Performance dashboards for Accuracy, F1, AUC, Forgetting Rate, and PR-AUC  

---

## 🧰 Tech Stack
**Languages & Frameworks:** Python | TensorFlow | PyTorch | Scikit-Learn  
**Libraries:** Pandas | NumPy | Matplotlib | Seaborn | LIME | SHAP  
**Environments:** Google Colab | Jupyter Notebook | Linux | Docker  

---

## 🚀 How to Use
1. Open any notebook in **Google Colab** or local Jupyter.  
2. Mount or upload dataset files (instructions inside each notebook).  
3. Run cells sequentially to train models, visualize results, and test robustness.  
4. Use provided functions for incremental learning and XAI analysis.  

```python
# Example: Training a CNN model on CIC-IDS-2018
from models.cnn import train_cnn
from utils.data_loader import load_cicids2018

X_train, X_test, y_train, y_test = load_cicids2018()
model = train_cnn(X_train, y_train, epochs=50, batch_size=64)

# Example: Running SHAP explainability
import shap
explainer = shap.Explainer(model, X_test[:100])
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test)

