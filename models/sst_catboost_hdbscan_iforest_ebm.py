import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor

from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import hdbscan
import warnings

warnings.filterwarnings("ignore")


def main():
    # -------------------- Load Dataset --------------------
    data = pd.read_csv("final_selected_features.csv")
    target_column = "class"

    # Drop identifier columns if present
    drop_cols = ['call_id', 'src_ip', 'dst_ip', 'from_uri', 'to_uri']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical features in X
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_features:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))

    # Encode labels
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    target_names = [str(cls) for cls in le_y.classes_]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------- SST Autoencoder Embedding --------------------
    autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 32, 64, 128), max_iter=100, random_state=42)
    autoencoder.fit(X_scaled, X_scaled)
    embeddings = autoencoder.predict(X_scaled)

    # -------------------- HDBSCAN --------------------
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(embeddings)

    # -------------------- Isolation Forest --------------------
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_scores = iso_forest.fit_predict(embeddings)

    # -------------------- Combine Features --------------------
    X_combined = pd.DataFrame(embeddings)
    X_combined["cluster"] = cluster_labels
    X_combined["outlier_score"] = outlier_scores
    X_combined["label"] = y_encoded

    X_final = X_combined.drop(columns=["label"])
    y_final = X_combined["label"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, stratify=y_final, random_state=42)

    # -------------------- CatBoost Classifier --------------------
    cat_model = CatBoostClassifier(verbose=0, random_state=42)
    cat_model.fit(X_train, y_train)
    cat_preds = cat_model.predict(X_test)

    print("\n=== ✅ SST + HDBSCAN + IF + CatBoost Results ===")
    print(f"\n✅ Accuracy: {accuracy_score(y_test, cat_preds):.4f}")
    print(f"✅ F1 Score: {f1_score(y_test, cat_preds, average='weighted'):.4f}")
    print("\n📋 Classification Report:\n", classification_report(y_test, cat_preds, target_names=target_names))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, cat_preds))

    # -------------------- EBM Classifier --------------------
    ebm_model = ExplainableBoostingClassifier(random_state=42)
    ebm_model.fit(X_train, y_train)
    ebm_preds = ebm_model.predict(X_test)

    print("\n=== ✅ SST + HDBSCAN + IF + EBM Results ===")
    print(f"\n✅ Accuracy: {accuracy_score(y_test, ebm_preds):.4f}")
    print(f"✅ F1 Score: {f1_score(y_test, ebm_preds, average='weighted'):.4f}")
    print("\n📋 Classification Report:\n", classification_report(y_test, ebm_preds, target_names=target_names))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, ebm_preds))


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
