import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

# -------------------- Voting Ensemble --------------------
class VotingEnsemble:
    def __init__(self, clf1, clf2):
        self.clf1 = clf1
        self.clf2 = clf2

    def predict(self, X):
        pred1 = self.clf1.predict(X).flatten()
        pred2 = self.clf2.predict(X).flatten()
        
        combined_preds = []
        for p1, p2 in zip(pred1, pred2):
            votes = [int(p1), int(p2)]
            combined_preds.append(np.bincount(votes).argmax())
        return np.array(combined_preds)

# -------------------- Main Script --------------------
def main():
    # Load data
    df = pd.read_csv("final_selected_features1.csv")
    target_column = "class"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------- SST via Autoencoder --------------------
    autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 32, 64, 128), max_iter=100, random_state=42)
    autoencoder.fit(X_scaled, X_scaled)
    embeddings = autoencoder.predict(X_scaled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # -------------------- Train Models --------------------
    catboost_model = CatBoostClassifier(verbose=0, random_state=42)
    ebm_model = ExplainableBoostingClassifier(random_state=42)

    catboost_model.fit(X_train, y_train)
    ebm_model.fit(X_train, y_train)

    # -------------------- Ensemble Prediction --------------------
    ensemble = VotingEnsemble(catboost_model, ebm_model)
    y_pred = ensemble.predict(X_test)

    # -------------------- Evaluation --------------------
    print("\n=== ✅ SST + CatBoost + EBM Ensemble Results ===")
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("✅ F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
