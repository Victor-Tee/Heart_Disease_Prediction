# Heart_Disease_Prediction
A machine learning project that predicts the likelihood of heart disease using classification models, feature engineering, and model interpretability techniques.


## Heart Disease Risk Prediction using Machine Learning
 ### 1. Project Overview
This project focuses on building, optimizing, and interpreting a Machine Learning model to predict a patient's risk of developing heart disease. This is a Binary Classification task crucial for clinical decision-making and early patient intervention.

Goal: To classify patients into two groups: High Risk (1) or Low Risk (0).

Business/Clinical Impact: In a high-stakes environment like healthcare, minimizing False Negatives (missed diagnoses) is critical. Therefore, the model optimization prioritizes high Recall (Sensitivity) and ROC-AUC scores.

### 2. Problem Statement & Motivation
Cardiovascular disease remains a leading cause of death globally. Accurate and early risk stratification can allow healthcare providers to implement preventive measures, personalize care plans, and improve patient outcomes.

The challenge lies in managing a dataset that can be noisy, contain categorical features, and requires a model that is not only accurate but also interpretable—doctors need to understand why a decision was made.

### 3. Dataset and Key Features
The dataset used is the well-known Heart Disease Dataset sourced from the UCI Machine Learning Repository (commonly found on Kaggle). It contains 14 features collected from patients, including demographic, clinical, and lifestyle data.

Feature Name,Type,Description
age,Numerical,Age in years.
sex,Categorical,(1 = male; 0 = female)
cp,Categorical,Chest pain type (4 values)
trestbps,Numerical,Resting blood pressure (in mm Hg)
chol,Numerical,Serum cholesterol in mg/dl
thalach,Numerical,Maximum heart rate achieved
exang,Categorical,Exercise induced angina (1 = yes; 0 = no)
oldpeak,Numerical,ST depression induced by exercise relative to rest
target,Target,Presence of heart disease (1) or not (0)

### 4. Methodology and Technical Process
The project followed a robust data science workflow, focusing on intermediate-level techniques in preprocessing and modeling.

#### A. Data Wrangling & Exploratory Data Analysis (EDA)
Initial Cleaning: Addressed data quality issues, specifically replacing biologically implausible zero values in numerical features like cholesterol and resting blood pressure with the median of their respective columns.

Key Insights:

Patients with atypical chest pain types (cp > 0) showed a significantly higher incidence of heart disease compared to those with asymptomatic chest pain.

A high negative correlation was observed between the target variable and thalach (max heart rate), suggesting individuals who can achieve a higher heart rate during exercise are less likely to have heart disease.

#### B. Preprocessing and Feature Engineering
One-Hot Encoding: Applied One-Hot Encoding to nominal categorical features (cp, thal, slope) to convert them into a machine-readable binary format.

Feature Scaling: Used StandardScaler to normalize all continuous numerical features (age, chol, trestbps, etc.). This prevents features with larger numerical scales from unfairly dominating the model's objective function.

Train-Test Split: The dataset was split into 80% training and 20% testing data, ensuring stratification (stratify=y) to maintain the class balance in both sets. 
<img width="916" height="834" alt="heatmap" src="https://github.com/user-attachments/assets/abc5a345-317a-4c44-9145-41e0cbd5eeec" />

#### C. Model Selection and Hyperparameter Tuning
Three models were tested: Logistic Regression (as a baseline), Random Forest, and XGBoost.

Model of Choice: XGBoost (Extreme Gradient Boosting) consistently achieved the highest discriminative power.

Optimization: Hyperparameter tuning was performed using GridSearchCV with 5-fold Cross-Validation on the training data. Crucially, the optimization score was set to maximize the Recall and ROC-AUC.
<img width="695" height="545" alt="boxplot" src="https://github.com/user-attachments/assets/74ac9366-75c9-431f-a908-966740a3cd1f" />


 Sample tuning code (demonstrative)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, make_scorer

 Prioritizing Recall for clinical relevance
recall_scorer = make_scorer(recall_score)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}

 GridSearchCV optimized for Recall
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring=recall_scorer)
grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_


### 5. Results and Evaluation
The final, optimized XGBoost model was evaluated on the unseen test data.

Metric,Score,Clinical Interpretation
ROC-AUC,0.89,"The model has strong discriminative power, correctly ranking a randomly chosen positive case higher than a negative case 89% of the time."
Recall (Sensitivity),0.85,The model successfully identified 85% of all patients who actually had heart disease (minimizing dangerous False Negatives).
Precision,0.81,"When the model predicts a patient has heart disease, it is correct 81% of the time (managing False Positives and unnecessary patient stress)."
Accuracy,0.84,The overall correctness of the model.


### 6. Model Interpretation with SHAP (The Differentiator)
To provide the transparency required in a clinical setting, the SHAP (SHapley Additive exPlanations) library was used to interpret the model's output.

Global Feature Importance: SHAP analysis revealed that the top three features driving the model's prediction were:

cp (Chest Pain Type)

thal (Thalassemia)

exang (Exercise-induced Angina)

Local Interpretability: SHAP allows for the explanation of individual patient predictions, showing which specific feature values pushed a patient towards a high-risk or low-risk outcome, providing direct insight for the clinician.
<img width="695" height="545" alt="histogram" src="https://github.com/user-attachments/assets/63b60378-4a9d-4b35-9d5d-29b4f8b59d97" />

### 7. Conclusion and Next Steps
The project successfully built an interpretable machine learning pipeline that can predict heart disease risk with a high degree of confidence and, critically, high Recall.

Future Steps:

Deployment: Deploying the model using a micro-framework (like Flask or Streamlit) to simulate a clinical decision support tool.

Temporal Data: Exploring models that can handle time-series data if patient visits/vitals over multiple years become available.

Fairness Analysis: Analyzing model performance across demographic groups (sex, age) to ensure equitable predictions.

### 8. Technologies Used
Python 3.x

Pandas & NumPy

Scikit-learn (for modeling and preprocessing)

XGBoost

Matplotlib & Seaborn (for EDA and visualization)

SHAP (for model interpretability)
