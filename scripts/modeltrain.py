import sklearn
import streamlit as st
import pandas as pd
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

print("Sklearn version:", sklearn.__version__)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\yashr\OneDrive\Desktop\loan approval\Loan_Data_Cleaned.csv")

# Features & target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# AdaBoost with deeper base tree
# -------------------------------
base_tree = DecisionTreeClassifier(max_depth=3, random_state=42)  # deeper tree
ada = AdaBoostClassifier(estimator=base_tree, random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0]
}

# Grid search
grid = GridSearchCV(ada, param_grid, scoring='accuracy', cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# -------------------------------
# Evaluate
# -------------------------------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Best AdaBoost Accuracy: {acc:.4f}")

# -------------------------------
# Feature importance
# -------------------------------
importances = best_model.feature_importances_
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(fi_df)

# -------------------------------
# Save model
# -------------------------------
with open("ab_best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nAdaBoost model saved as ab_best_model.pkl")
