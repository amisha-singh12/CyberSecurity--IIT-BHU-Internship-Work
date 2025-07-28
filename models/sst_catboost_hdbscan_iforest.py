
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report

from catboost import CatBoostClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import hdbscan

# -------------------- Load Dataset --------------------
data = pd.read_csv("final_selected_features.csv")  # Replace with your actual file
target_column = "class"

X = data.drop(columns=[target_column])
y = data[target_column]

# Encode categorical features for Autoencoder + CatBoost
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Normalize for autoencoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- Self-Supervised Tabular Model (SST - Simulated via Autoencoder) --------------------
# Use Autoencoder to extract latent representations
autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 32, 64, 128), max_iter=100, random_state=42)
autoencoder.fit(X_scaled, X_scaled)
embeddings = autoencoder.predict(X_scaled)

# -------------------- HDBSCAN --------------------
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(embeddings)

# -------------------- Isolation Forest --------------------
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_scores = iso_forest.fit_predict(embeddings)

# -------------------- Combine features --------------------
X_combined = pd.DataFrame(embeddings)
X_combined["cluster"] = cluster_labels
X_combined["outlier_score"] = outlier_scores
X_combined["label"] = y.values

# Train/Test Split
X_final = X_combined.drop(columns=["label"])
y_final = X_combined["label"]
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, stratify=y_final, random_state=42)

# -------------------- CatBoost Classifier --------------------
model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------- Results --------------------
print("=== Final Classification Report ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
