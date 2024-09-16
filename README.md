# ML-Project-Deposit-Subscription-Prediction

**Project Description**

This project focuses on predicting whether a client will subscribe to a term deposit based on a set of features related to their demographics, economic factors, and past interactions with the bank. The dataset contains both categorical and numerical variables such as age, job, marital status, contact type, and duration of the last call. The task is a binary classification problem where the target variable y indicates whether the client has subscribed to the term deposit (yes or no).

**Task Overview**

The main objective of the project is to build machine learning models that can accurately predict client subscription to a term deposit. The challenge lies in handling class imbalance, where the majority of the clients do not subscribe. Multiple models were experimented with to optimize performance based on F1 score.

**Models and Approaches**

Several machine learning models were applied to this task, including Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree Classifier, XGBoost Classifier.

**Experimental Results**

| Model Name              | Train Score | Validation Score |
|-------------------------|-------------|------------------|
| LogisticRegression       | 0.47        | 0.49             |
| KNeighborsClassifier     | 0.5         | 0.5              |
| DecisionTreeClassifier   | 0.5         | 0.52             |
| XGBClassifier (model 1)  | 0.51        | 0.53             |
| XGBClassifier (model 2)  | 0.54        | 0.53             |

**Conclusions**

**Best Model:** The XGBoost models performed best, achieving a validation F1 score of 0.53.  

**Challenges:** The F1 score is relatively low, indicating that the dataset might need more features or better feature engineering to capture the patterns that lead to subscription.

**Future Work**

**Feature Engineering:** Additional data about customer behavior or previous campaign details could improve performance.

**Handling Imbalance:** Undersampling or oversampling could be explored.

**Explore Other Models:** Deep learning models or more complex ensemble methods might yield better results.
