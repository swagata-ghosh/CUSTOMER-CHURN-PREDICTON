
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, make_scorer, f1_score, recall_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import optuna

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Drop irrelevant columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Features and labels
X = df.drop('Exited', axis=1)
y = df['Exited']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Custom scorer

def custom_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    if recall < 0.65:
        return 0  # Reject models with recall below threshold
    return f1_score(y_true, y_pred)

scorer = make_scorer(custom_score, greater_is_better=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': 1
    }

    model = XGBClassifier(
        **params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, scoring=scorer, cv=skf)
    return scores.mean()

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Best model training
best_params = study.best_params
final_model = XGBClassifier(
    **best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
final_model.fit(X_train, y_train)

# Evaluation
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 4))
print("\nBest Parameters:", best_params)