import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your CSV file
df = pd.read_csv("final_selected_features1.csv")  # Update path if needed

# Drop non-predictive or unique identifiers (if present)
drop_cols = ['call_id', 'src_ip', 'dst_ip', 'from_uri', 'to_uri']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Encode target (attack class)
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Identify boolean categorical features (CatBoost handles these)
cat_features = df.select_dtypes(include=['bool']).columns.tolist()

# Separate features and target
X = df.drop(columns=['class'])
y = df['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define CatBoost Pools
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

# Initialize and train the CatBoost model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=100
)

model.fit(train_pool, eval_set=test_pool)

# Predict and squeeze to flatten output
y_pred = model.predict(test_pool).squeeze()

# Convert target names to string for classification report
target_names = [str(cls) for cls in label_encoder.classes_]

# Evaluation Metrics
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and label encoder
model.save_model("catboost_voip_model.cbm")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n💾 Model and encoder saved.")

