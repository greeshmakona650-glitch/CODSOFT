import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD DATA
data = pd.read_csv("Churn_Modelling.csv")

data.columns = data.columns.str.strip()

# 2. CLEAN DATA
cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

# 3. TARGET
target_col = "Exited"

# 4. SPLIT FEATURES & TARGET
X = data.drop(target_col, axis=1)
y = data[target_col]

# 5. ENCODING
X = pd.get_dummies(X, drop_first=True)

# 6. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 7. FEATURE SCALING (IMPORTANT)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. BALANCE DATA (SMOTE)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", y_train.value_counts())

# 9. MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier()
}

# 10. TRAIN + EVALUATE

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    results[name] = acc

    print("\n=========================")
    print(name)
    print("Accuracy:", acc)
    print("ROC-AUC:", roc)
    print(classification_report(y_test, y_pred))

# 11. BEST MODEL (simple comparison)
best_model = max(results, key=results.get)
print("\n🏆 Best Model:", best_model)

# 12. CONFUSION MATRIX (BEST MODEL)
final_model = RandomForestClassifier(n_estimators=200, random_state=42)
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Best Model (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 13. FEATURE IMPORTANCE
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n🔥 Top Important Features:\n")
print(importance.head(10))