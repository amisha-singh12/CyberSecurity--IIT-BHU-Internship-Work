import pandas as pd
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from models.catboost_model import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import hdbscan

# ========== Load Dataset ==========
df = pd.read_csv("final_selected_features.csv")

if "class" not in df.columns:
    raise ValueError("No 'class' column found. Supervised training requires labels.")

X = df.drop(columns=["class"])
y = df["class"].astype(int)

# ========== Drop Rare Classes ==========
label_counts = y.value_counts()
valid_labels = label_counts[label_counts >= 2].index
X = X[y.isin(valid_labels)]
y = y[y.isin(valid_labels)]

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

if len(y_train.unique()) < 2:
    raise ValueError("Training set has less than 2 classes. Please review dataset.")

# ========== Model 1: CatBoost ==========
catboost = CatBoostClassifier(verbose=0, random_state=42)
catboost.fit(X_train, y_train)
pred_cb = catboost.predict(X_test)

# ========== Model 2: TabNet ==========
tabnet = TabNetClassifier(verbose=0)
tabnet.fit(X_train.values, y_train.values, max_epochs=100)
pred_tabnet = tabnet.predict(X_test.values)

# ========== Model 3: EBM ==========
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
pred_ebm = ebm.predict(X_test)

# ========== Model 4: HDBSCAN + Isolation Forest ==========
# HDBSCAN to cluster full dataset (unsupervised)
clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
cluster_labels = clusterer.fit_predict(X)

# Isolation Forest to detect outliers
iso_forest = IsolationForest(contamination=0.15, random_state=42)
iso_forest.fit(X_train)
outlier_scores = iso_forest.predict(X_test)
# Convert: -1 (anomaly) → 1 (attack), 1 → 0 (normal)
outlier_scores = np.where(outlier_scores == -1, 1, 0)

# ========== Model 5: Self-Supervised Placeholder (Logistic Regression) ==========
ssl_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
ssl_model.fit(X_train, y_train)
pred_ssl = ssl_model.predict(X_test)

# ========== Majority Voting ==========
predictions = np.array([
    pred_cb,
    pred_tabnet.reshape(-1),
    pred_ebm,
    outlier_scores,
    pred_ssl
])

final_preds = []
for i in range(predictions.shape[1]):
    vote = np.bincount(predictions[:, i]).argmax()
    final_preds.append(vote)

# ========== Evaluation ==========
start_time = time.time()

accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds, average="weighted", zero_division=0)
recall = recall_score(y_test, final_preds, average="weighted")
f1 = f1_score(y_test, final_preds, average="weighted")

end_time = time.time()
prediction_time = end_time - start_time

print(f"\nTest Set Metrics:")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Prediction Time: {prediction_time:.3f} seconds")

print("\nClassification Report:")
print(classification_report(y_test, final_preds, zero_division=0))