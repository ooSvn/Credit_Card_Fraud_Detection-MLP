# Credit Card Fraud Detection using MLP

This project focuses on building and evaluating **Multi-Layer Perceptron (MLP)** neural networks to detect fraudulent credit card transactions. The work was done as part of the *Neural Networks and Deep Learning* course homework.

## 📌 Project Description

The task involves:

* Exploring and preprocessing the **Credit Card Fraud Detection dataset**.
* Designing, implementing, and training MLP models with different depths and regularization techniques.
* Evaluating performance using multiple metrics beyond accuracy, due to dataset imbalance.
* Performing hyperparameter tuning (Grid Search).
* Comparing MLP models with **Logistic Regression** as a baseline.

## ⚙️ Tasks Overview

### 1. Data Preprocessing & Exploration

* Downloaded dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
* Displayed dataset summary.
* Plotted **class distribution bar chart** to highlight imbalance between fraudulent and non-fraudulent transactions.
* Explained challenges of imbalanced datasets.
* Normalized features using `StandardScaler` / `MinMaxScaler`.
* Split dataset into **70% training** and **30% testing**, ensuring class balance.

### 2. Simple MLP Model

* Architecture:

  * Input layer: number of features = dataset dimensions.
  * Hidden layer: **64 neurons (ReLU)**.
  * With and without **Dropout (30%)**.
  * With and without **L2 regularization (λ = 0.0001)**.
  * Output layer: single neuron with **sigmoid** activation.
* Training:

  * Loss: Binary Cross-Entropy.
  * Optimizer: Adam.
  * Batch size = 32.
  * Epochs = 40.
* Evaluation:

  * Plotted training **loss** and **accuracy curves**.
  * Computed metrics: **Confusion Matrix, Accuracy, Precision, Recall, F1-score, ROC-AUC**.

### 3. Deeper MLP Model

* Architecture:

  * Hidden Layer 1: 128 neurons (ReLU).
  * Hidden Layer 2: 64 neurons (ReLU).
  * Dropout: 20% after each hidden layer.
  * L2 regularization (λ = 0.0001).
* Training and evaluation as in the previous step.
* Compared results with the simpler MLP.

### 4. Confusion Matrix & Metrics Analysis

* Discussed what each metric (Accuracy, Precision, Recall, F1-score, ROC, AUC) measures.
* Analyzed why **accuracy alone is misleading** on imbalanced data.
* Identified which class was more frequently misclassified.
* Discussed **trade-offs** between Precision and Recall.

### 5. Hyperparameter Tuning (Grid Search)

* Performed grid search over:

  * Hidden layer size: {64, 128, 256}
  * Dropout: {0.2, 0.3, 0.4}
  * L2 regularization: {0.0001, 0.001}
  * Batch size: {16, 32, 64}
* Selected the **best-performing combination** and retrained the model.
* Evaluated with the same metrics as before.

### 6. Logistic Regression Baseline

* Trained a **Logistic Regression** model on the same data.
* Compared its performance with the best MLP model.
* Analyzed conditions where Logistic Regression may outperform deep models.

### 7. Final Analysis & Summary

* Identified the best-performing model and explained why.
* Compared shallow vs. deep MLP models.
* Analyzed impact of hyperparameter tuning.
* Highlighted **error patterns** in confusion matrices.
* Discussed Precision-Recall trade-off in fraud detection.
* Suggested potential improvements (e.g., SMOTE, anomaly detection, ensemble methods).
* Reflected on real-world fraud detection challenges.

## 📊 Example Results

* The **deeper MLP with tuned hyperparameters** achieved the best balance between Recall and Precision.
* **Logistic Regression** provided competitive performance but underperformed in Recall compared to MLP.
* Accuracy was not a reliable metric due to heavy class imbalance.
* **ROC-AUC and F1-score** proved to be better indicators of performance.

## 🧠 Key Takeaways

* Fraud detection requires models that prioritize **Recall** (catching fraud) over simple accuracy.
* Deep networks (MLPs) can outperform simpler models when tuned properly, but they require more computation.
* Regularization and Dropout play a key role in preventing overfitting.
* Hyperparameter optimization significantly improves results.
