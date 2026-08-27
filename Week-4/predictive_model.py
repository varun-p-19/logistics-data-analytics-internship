import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

FILE_NAME = "cleaned_logistics_data.csv"

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        f"'{FILE_NAME}' was not found. Place the cleaned Week 2 "
        "CSV file in the same folder as this script."
    )

df = pd.read_csv(FILE_NAME)

print("=" * 70)
print("WEEK 4 - PREDICTIVE MODELING AND OPTIMIZATION")
print("=" * 70)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

if "order_purchase_timestamp" in df.columns:
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce"
    )

    df["purchase_month"] = (
        df["order_purchase_timestamp"].dt.month
    )

    df["purchase_dayofweek"] = (
        df["order_purchase_timestamp"].dt.dayofweek
    )

if "is_late" not in df.columns:

    required_dates = {
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    }

    if required_dates.issubset(df.columns):

        df["order_delivered_customer_date"] = pd.to_datetime(
            df["order_delivered_customer_date"],
            errors="coerce"
        )

        df["order_estimated_delivery_date"] = pd.to_datetime(
            df["order_estimated_delivery_date"],
            errors="coerce"
        )

        df["is_late"] = (
            df["order_delivered_customer_date"]
            > df["order_estimated_delivery_date"]
        ).astype(int)

    else:
        raise ValueError(
            "Target column 'is_late' is missing and the required "
            "delivery date columns were not found."
        )


df = df.dropna(subset=["is_late"]).copy()

df["is_late"] = df["is_late"].astype(int)

print("\nTarget distribution:")
print(df["is_late"].value_counts())
print("\nTarget percentage:")
print(df["is_late"].value_counts(normalize=True) * 100)

candidate_numeric = [
    "order_value",
    "freight_value",
    "number_of_items",
    "purchase_month",
    "purchase_dayofweek",
]

candidate_categorical = [
    "customer_state",
]

numeric_features = [
    col for col in candidate_numeric
    if col in df.columns
]

categorical_features = [
    col for col in candidate_categorical
    if col in df.columns
]

feature_columns = numeric_features + categorical_features

if not feature_columns:
    raise ValueError(
        "No expected model features were found in the dataset."
    )

X = df[feature_columns].copy()
y = df["is_late"].copy()

print("\nFeatures used:")
print(feature_columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    ),
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    ),
])

logistic_model.fit(X_train, y_train)

log_predictions = logistic_model.predict(X_test)
log_probabilities = logistic_model.predict_proba(X_test)[:, 1]

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    ),
])

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

def evaluate_model(name, y_true, predictions, probabilities):
    """Print common classification metrics."""

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
    }


log_results = evaluate_model(
    "Logistic Regression",
    y_test,
    log_predictions,
    log_probabilities
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
)

results = pd.DataFrame([
    log_results,
    rf_results
])

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(results.round(4))

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    random_forest_model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

print("F1 scores:", np.round(cv_scores, 4))
print("Mean CV F1:", round(cv_scores.mean(), 4))
print("Standard deviation:", round(cv_scores.std(), 4))

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

parameter_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [8, 12, 16, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}

search = RandomizedSearchCV(
    estimator=random_forest_model,
    param_distributions=parameter_grid,
    n_iter=10,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)

print("Best parameters:")
print(search.best_params_)

print("\nBest cross-validation F1:")
print(round(search.best_score_, 4))

best_model = search.best_estimator_

best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

best_results = evaluate_model(
    "Tuned Random Forest",
    y_test,
    best_predictions,
    best_probabilities
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

try:
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_classifier = best_model.named_steps["classifier"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    importance = pd.Series(
        fitted_classifier.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print("\nTop 15 important features:")
    print(importance.head(15))

    importance.head(15).to_csv(
        "feature_importance.csv",
        header=["importance"]
    )

except Exception as error:
    print(
        "Feature importance could not be extracted:",
        error
    )

risk_output = X_test.copy()

risk_output["actual_is_late"] = y_test.values
risk_output["delay_probability"] = best_probabilities

risk_output["risk_level"] = pd.cut(
    risk_output["delay_probability"],
    bins=[0, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

risk_output = risk_output.sort_values(
    "delay_probability",
    ascending=False
)

risk_output.to_csv(
    "predicted_delay_risk.csv",
    index=False
)

print("\nTop high-risk orders:")
print(risk_output.head(10))

print("\n" + "=" * 70)
print("LOGISTICS OPTIMIZATION SUMMARY")
print("=" * 70)

high_risk_count = (
    risk_output["risk_level"]
    .eq("High")
    .sum()
)

medium_risk_count = (
    risk_output["risk_level"]
    .eq("Medium")
    .sum()
)

low_risk_count = (
    risk_output["risk_level"]
    .eq("Low")
    .sum()
)

print(f"High-risk shipments  : {high_risk_count}")
print(f"Medium-risk shipments: {medium_risk_count}")
print(f"Low-risk shipments   : {low_risk_count}")

print("""
Suggested operational actions:

1. High-risk shipments:
   - Prioritize for monitoring.
   - Consider additional delivery capacity.
   - Contact the customer proactively when appropriate.

2. Medium-risk shipments:
   - Monitor progress closely.
   - Review carrier and regional performance.

3. Low-risk shipments:
   - Continue through the normal logistics process.

The thresholds used here (30% and 60%) are illustrative.
In a real deployment, thresholds should be selected using
validation results and the cost of different operational actions.
""")

all_results = pd.DataFrame([
    log_results,
    rf_results,
    best_results
])

all_results.to_csv(
    "model_evaluation_results.csv",
    index=False
)

print("\nFiles generated:")
print("- model_evaluation_results.csv")
print("- feature_importance.csv (if feature extraction succeeds)")
print("- predicted_delay_risk.csv")

print("\n" + "=" * 70)
print("WEEK 4 PREDICTIVE MODELING COMPLETED")
print("=" * 70)
