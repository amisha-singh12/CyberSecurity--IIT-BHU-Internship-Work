import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    # Load the dataset
    df = pd.read_csv("final_selected_features1.csv")

    # Drop non-useful identifier columns
    drop_cols = ['call_id', 'src_ip', 'dst_ip', 'from_uri', 'to_uri']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Encode target
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])

    # Convert boolean to int if needed
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)

    # Prepare features and labels
    X = df.drop(columns=['class'])
    y = df['class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train EBM
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X_train, y_train)

    # Predict
    y_pred = ebm.predict(X_test)

    # Evaluation
    target_names = [str(c) for c in le.classes_]
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("✅ F1 Score:", f1_score(y_test, y_pred, average="weighted"))
    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and encoder
    joblib.dump(ebm, "ebm_voip_model.pkl")
    joblib.dump(le, "label_encoder_ebm.pkl")
    print("\n💾 Model and label encoder saved.")

if __name__ == "__main__":
    main()
