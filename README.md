# Credit Card Fraud Anomaly Detection

An anomaly detection system for identifying suspicious credit card transactions using both supervised(random forest classifier) and unsupervised(isolation anomaly detector) machine learning.

The project compares both and implements basic drift detection to identify retraining.

## Project Objectives

* Detect fraudulent credit card transactions.
* Handle severe class imbalance.
* Compare supervised and unsupervised detection approaches.
* Evaluate models using Precision-Recall AUC (PR-AUC).
* Analyze the effect of different fraud probability thresholds.
* Detect changes in feature distributions using the Kolmogorov-Smirnov (KS) test.
* Generate a retraining alert when detected drift exceeds a predefined threshold.

## Dataset

This project uses the Credit Card Fraud Detection dataset.

Before running the notebook, download the dataset and upload creditcard.csv into the /content/  directory in Colab.

The dataset classifies transactions as:

* 0 — legitimate transaction
* 1 — fraudulent transaction

## Requirements

At least Python 3.X is needed with the following libraries:

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Matplotlib
* Google Colab / Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/credit-card-fraud-anomaly-detection.git
cd credit-card-fraud-anomaly-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Google Colab

1. Open the notebook:

```text
notebooks/Credit_Card_Fraud_Anomaly_Detection.ipynb
```

2. Open it in Google Colab.
3. Upload `creditcard.csv` to the Colab `/content/` directory.
4. Run the notebook cells from top to bottom (or just run all).

The notebook performs:

1. Cleaning and loading the dataset into Colab
2. Checks data and fraud distribution and returns a graph
3. Train/test splitting
4. Random Forest training
5. Random Forest evaluation
6. Isolation Forest training and evaluation
7. Threshold analysis
8. Drift detection
9. Creates a final detection function as a whole
10. Tests the function using a piece of the dataset

The final output should show you DRIFT DETECTED or NO SIGNIFICANT DRIFT as well as whether a given transaction is fraud or not.

Check the `/docs/PROJECT_DESCRIPTION.md` for the complete explanation of the code, as well as key design decisions. 


