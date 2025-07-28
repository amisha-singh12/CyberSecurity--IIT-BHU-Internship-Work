import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")  # Optional: Suppress sklearn warnings


def main():
    # -------------------- Load Dataset --------------------
    data = pd.read_csv("final_selected_features.csv")  # Replace with your dataset path
    target_column = "class"

    # Drop identifier columns if present
    drop_cols = ['call_id', 'src_ip', 'dst_ip', 'from_uri', 'to_uri']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # Separate features and labels
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical features in X (if any)
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_features:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))

    # Encode labels (y)
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    target_names = [str(cls) for cls in le_y.classes_]  # 🔧 Convert to string for sklearn

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------- Self-Supervised Tabular (SST) via Autoencoder --------------------
    autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 32, 64, 128), max_iter=100, random_state=42)
    autoencoder.fit(X_scaled, X_scaled)
    embeddings = autoencoder.predict(X_scaled)

    # -------------------- Train/Test Split --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # -------------------- CatBoost Classifier --------------------
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -------------------- Evaluation --------------------
    print("\n=== ✅ SST + CatBoost Results ===")
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
