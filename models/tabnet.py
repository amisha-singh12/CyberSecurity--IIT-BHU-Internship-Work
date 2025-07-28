import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

# Load your dataset
df = pd.read_csv("your_selected_feature_dataset.csv")  # Replace with actual filename

# Drop non-useful columns
drop_cols = ['call_id', 'src_ip', 'dst_ip', 'from_uri', 'to_uri']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Encode target class
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# Handle boolean columns
bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
df[bool_cols] = df[bool_cols].astype(int)

# Split data
X = df.drop(columns=['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42, stratify=y
)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to float32 for TabNet
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Train TabNet Classifier
tabnet = TabNetClassifier(
    verbose=1,
    seed=42
)

tabnet.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=200,
    patience=20,
    batch_size=128,
    virtual_batch_size=64
)

# Predict
y_pred = tabnet.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model + label encoder
tabnet.save_model("tabnet_voip_model")
import joblib
joblib.dump(le, "label_encoder_tabnet.pkl")
