# Credit Card Fraud Anomaly Detection

## 1. Overview

This project develops a fraud detection system for identifying suspicious credit card transactions. 

Two independent machine learning approaches are implemented and compared:

* **Random Forest** — supervised learning using the available fraud labels.
* **Isolation Forest** — unsupervised anomaly detection without using fraud labels during training.

The project also performs fraud probability threshold analysis and basic feature drift detection to determine when model retraining may be required.

---

## 2. Data Preparation

The Credit Card Fraud Detection dataset contains transaction features and a binary `Class` label, where `0` represents a legitimate transaction and `1` represents fraud.

The dataset was loaded and checked for invalid and missing values. Rows containing missing values were removed before training. The data was then divided into training and test sets using a stratified 80:20 split with `random_state=42`.

Stratification was used to keep the fraud ratio similar for both sets.
---

## 3. Handling Class Imbalance

In the given Kaggle dataset, the fraud ratio was disproportionately low compared to legitimate transactions, requiring the model to handle class imbalance.

Random Forest was trained using:

```python
class_weight="balanced"
```
Class weighting was used to handle the class imbalance. This gives more importance to the fraud class during training instead of just naive oversampling.

---

## 4. Supervised Model — Random Forest

Random Forest was trained using the given fraud labels in the dataset.

Using the default probability threshold of 0.50, the model achieved:

| Metric    |     Result |
| --------- | ---------: |
| Precision |     96.05% |
| Recall    |     74.49% |
| F1-score  |     83.91% |
| PR-AUC    | **85.42%** |

The confusion matrix was:

```text
[[56861     3]
 [   25    73]]
```

This corresponds to 73 correctly detected fraudulent transactions, 3 false positives, and 25 missed fraudulent transactions.

---

## 5. Unsupervised Model — Isolation Forest

Here, the labels were not used for training. After training, its anomaly scores were compared with the known fraud labels for evaluation.

Isolation Forest achieved:

| Model            |     PR-AUC |
| ---------------- | ---------: |
| Isolation Forest | **21.80%** |

The result is substantially lower than the Random Forest PR-AUC of 85.42%.

This shows that not all statistical outliers can be considered as fraudulent. However, Isolation Forest is still valuable for comparison.

The two models are used purely for comparison, and the final detection only uses Random Forest since it achieved a better result.
---

## 6. Why use Precision-Recall AUC?

Since the dataset has a very small proportion of fradulent transactions, high accuracies can be obtained by just always predicting the majority class.

Since the model should aim to detect fradulent transactions with similar precision, PR-AUC is prefered.

PR-AUC focuses on precision and recall for the minority fraud class, making it more useful to compare detection than others.

The model comparison was:

| Model            |     PR-AUC |
| ---------------- | ---------: |
| Random Forest    | **85.42%** |
| Isolation Forest | **21.80%** |

Random Forest clearly performs better on this labelled dataset.

---

## 7. Threshold Analysis

The Random Forest probability threshold was varied from 0.1 to 0.9 to check which one gave the best F1-Score.

The results showed that lowering the threshold generally increases recall but can also increase false positives. The system has to consider whether the system would rather incorrectly reject a real transaction, or accidentally let a fradulent one through.

A threshold of **0.40** was selected for the current operating point.

At this threshold:

| Metric          |     Result |
| --------------- | ---------: |
| Precision       | **95.18%** |
| Recall          | **80.61%** |
| F1-score        | **87.29%** |
| False Positives |      **4** |
| False Negatives |     **19** |

---
## 8. False Positive vs False Negative Cost

False Positive: legitimate transaction incorrectly flagged as fraud, which can cause customer inconvenience and unnecessary verification.
False Negative: fraudulent transaction that was missed, potentially resulting in financial loss and security impact. 

Since real monetary costs could not be analysed, qualitative cost comparison was used.

A threshold of 0.4 was selected because it provides a good balance between these two error types. At this threshold, the model achieves 95.18% precision and 80.61% recall, with only 4 false positives and 19 false negatives. 
Compared with the default threshold of 0.5, lowering the threshold to 0.4 adds only one false positive but detects six additional fraudulent transactions. 
This can be considered as an acceptable trade off between both sides (for the given data), but a real deployment would probably change the value based on real financial considerations.. 

---

## 9. Drift Detection

Drift Detection: checks if the data changes significantly over time, and whether the model needs to be retrained for those values.

A basic feature drift detection mechanism was implemented using the **Kolmogorov-Smirnov (KS) test**.

The data was divided chronologically into an earlier reference window and a later simulated window. Numerical transaction features were compared between the two windows.

`Time` and `Class` were excluded from drift monitoring. Time was excluded since the data was ordered chronologically, so a drift in time values is expected and can be ignored. Class doesn't need to be monitored since we have a random forest model for that.

A KS statistic of **0.10 or greater** was selected as the threshold for meaningful feature drift.

The system monitors 29 features (31 total - 2 exclusions).

The results were:

| Measurement          |                     Result |
| -------------------- | -------------------------: |
| Features monitored   |                         29 |
| Drifted features     |                         14 |
| Drift percentage     |                 **48.28%** |
| Retraining threshold |                        20% |
| Alert                | **Retraining recommended** |

Since 48.28% of monitored features exceeded the drift threshold, the system generated a retraining recommendation.

---

## 10. Key Decisions

The main design decisions were:

1. **Class weighting** was used to address the severe class imbalance while keeping the test set unchanged.
2. **Random Forest and Isolation Forest were kept independent** so that supervised and unsupervised approaches could be directly compared.
3. **PR-AUC** was selected as the primary comparison metric because fraud is a highly imbalanced class.
4. **Multiple probability thresholds** were evaluated instead of assuming that 0.50 is always optimal.
5. A **0.40 Random Forest threshold** was selected as the current operating point based on the precision-recall and false-positive/false-negative trade-off.
6. **KS-based feature drift detection** was implemented to identify changes between an earlier and later data window.
7. A **20% drift alert threshold** was used to trigger a retraining recommendation.

---

## 11. Bonus objective

A shifted data slice was injected to simulate drift, and the system was tested to check whether drift occured. 
The values V1, V2, and Amount were shifted to create a change in their distributions. The same KS-test-based monitoring procedure was then applied to the modified data. The injected changes increased the KS statistics for the affected features, allowing the monitoring system to identify the shifted data and evaluate the retraining alert condition.

---

## 12. Final Results

The supervised Random Forest model significantly outperformed the unsupervised Isolation Forest model, achieving a PR-AUC of 85.42% compared with 21.80%.

Threshold analysis showed that changing the Random Forest threshold from 0.50 to 0.40 improved recall from 74.49% to 80.61% while maintaining 95.18% precision.

The drift monitoring component detected meaningful distribution changes in 14 of 29 monitored features, exceeding the 20% alert threshold and resulting in a retraining recommendation.

Finally, the system was tested to check whether drift was properly detected and whether transaction validity check was accurate.

