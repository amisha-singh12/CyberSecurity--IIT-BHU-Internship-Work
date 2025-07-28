import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, classification_report

from interpret.glassbox import ExplainableBoostingClassifier

def main():
    # -------------------- Load Dataset --------------------
    data = pd.read_csv("final_selected_features.csv")
    target_column = "class"

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical features for Autoencoder
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------- Self-Supervised Tabular (SST) via Autoencoder --------------------
    autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 32, 64, 128), max_iter=100, random_state=42)
    autoencoder.fit(X_scaled, X_scaled)
    embeddings = autoencoder.predict(X_scaled)

    # -------------------- Train/Test Split --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, stratify=y, random_state=42
    )

    # -------------------- EBM Classifier --------------------
    ebm_model = ExplainableBoostingClassifier(random_state=42)
    ebm_model.fit(X_train, y_train)
    y_pred = ebm_model.predict(X_test)

    # -------------------- Results --------------------
    print("=== SST + EBM Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
