import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode

from models.catboost_model import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------- Load Dataset --------------------
data = pd.read_csv("final_selected_features.csv")  # Replace with your file
target_column = "class"

X = data.drop(columns=[target_column])
y = data[target_column]

# -------------------- Categorical Features --------------------
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_features:
    X[col] = X[col].astype("category")

# -------------------- Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------- CatBoost --------------------
print("\n=== CatBoost Training ===")
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train, cat_features=cat_features)
cat_preds = cat_model.predict(X_test)

# -------------------- TabNet --------------------
print("\n=== TabNet Training ===")
X_tab = X.copy()
for col in cat_features:
    le = LabelEncoder()
    X_tab[col] = le.fit_transform(X_tab[col].astype(str))

scaler = StandardScaler()
X_tab_scaled = pd.DataFrame(scaler.fit_transform(X_tab), columns=X_tab.columns)

X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X_tab_scaled, y, test_size=0.2, stratify=y, random_state=42
)

tabnet_model = TabNetClassifier(verbose=0)
tabnet_model.fit(
    X_train_tab.values, y_train_tab.values.reshape(-1, 1),
    eval_set=[(X_test_tab.values, y_test_tab.values.reshape(-1, 1))],
    eval_name=["val"]
)
tabnet_preds = tabnet_model.predict(X_test_tab.values)

# -------------------- EBM --------------------
print("\n=== EBM Training ===")
ebm_model = ExplainableBoostingClassifier()
ebm_model.fit(X_train, y_train)
ebm_preds = ebm_model.predict(X_test)

# -------------------- Ensemble: Majority Voting --------------------
# Ensure all predictions are in the same index order
combined_preds = np.array([
    cat_preds.astype(str),
    tabnet_preds.astype(str),
    ebm_preds.astype(str)
])

# Majority vote across the 3 models
final_preds = mode(combined_preds, axis=0, keepdims=False)[0]

# -------------------- Results --------------------
print("\n=== CatBoost Accuracy ===")
print(accuracy_score(y_test, cat_preds))
print(classification_report(y_test, cat_preds))

print("\n=== TabNet Accuracy ===")
print(accuracy_score(y_test_tab, tabnet_preds))
print(classification_report(y_test_tab, tabnet_preds))

print("\n=== EBM Accuracy ===")
print(accuracy_score(y_test, ebm_preds))
print(classification_report(y_test, ebm_preds))

print("\n=== Combined (Ensemble) Accuracy ===")
print(accuracy_score(y_test, final_preds))
print(classification_report(y_test, final_preds))
