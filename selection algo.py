import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, f_classif, VarianceThreshold, RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import warnings
warnings.filterwarnings("ignore")

# 🔹 Load dataset
df = pd.read_csv("combined_sip_features1.csv")

# 🔹 Drop ID column if exists
if "call_id" in df.columns:
    df.drop(columns=["call_id"], inplace=True)

# 🔹 Encode 'class' column
df['class'] = LabelEncoder().fit_transform(df['class'])

# 🔹 Auto-select numeric features except target
target_col = 'class'
feature_cols = [col for col in df.columns if col != target_col and pd.api.types.is_numeric_dtype(df[col])]

X = df[feature_cols]
y = df[target_col]

# 🔹 Train/Test Split (Fixed random state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔹 Initialize selection counter
feature_votes = {feature: 0 for feature in feature_cols}

print("\n✅ Running All Feature Selection Algorithms...\n")

# 1. Mutual Information (non-deterministic)
mi = SelectKBest(score_func=mutual_info_classif, k='all').fit(X_train, y_train)
mi_scores = dict(zip(feature_cols, mi.scores_))
top_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:3]
for f, _ in top_mi:
    feature_votes[f] += 1

# 2. ANOVA F-test (deterministic)
anova = SelectKBest(score_func=f_classif, k='all').fit(X_train, y_train)
anova_scores = dict(zip(feature_cols, anova.scores_))
top_anova = sorted(anova_scores.items(), key=lambda x: x[1], reverse=True)[:3]
for f, _ in top_anova:
    feature_votes[f] += 1

# 3. L1 Logistic Regression
l1_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
l1_model.fit(X_train, y_train)
for f, c in zip(feature_cols, l1_model.coef_[0]):
    if c != 0:
        feature_votes[f] += 1

# 4. Variance Threshold
vt = VarianceThreshold(threshold=0.1)
vt.fit(X_train)
vt_features = [feature_cols[i] for i in vt.get_support(indices=True)]
for f in vt_features:
    feature_votes[f] += 1

# 5. RFE
rfe_model = LogisticRegression(random_state=42)
rfe = RFE(estimator=rfe_model, n_features_to_select=3)
rfe.fit(X_train, y_train)
rfe_features = [f for f, s in zip(feature_cols, rfe.support_) if s]
for f in rfe_features:
    feature_votes[f] += 1

# 6. Forward Selection
fs_model = LogisticRegression(random_state=42)
sfs = SFS(fs_model, k_features=3, forward=True, floating=False, scoring='accuracy', cv=5)
sfs.fit(X_train, y_train)
forward_features = list(sfs.k_feature_names_)
for f in forward_features:
    feature_votes[f] += 1

# 7. Random Forest Feature Importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_scores = dict(zip(feature_cols, rf.feature_importances_))
top_rf = sorted(rf_scores.items(), key=lambda x: x[1], reverse=True)[:3]
for f, _ in top_rf:
    feature_votes[f] += 1

# 🔹 Display feature vote count (cleaned)
print("📊 Feature Occurrence Count (across algorithms):")
sorted_votes = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
for f, count in sorted_votes:
    print(f"{f:25}: {count} votes")

# 🔹 Choose final selected features (at least 3 votes)
final_features = [f for f, count in sorted_votes if count >= 3]
print("\n✅ Final Selected Features (3+ votes):", final_features)

# 🔹 Save new dataset with selected features + label
final_df = df[final_features + [target_col]]
final_df.to_csv("final_selected_features1.csv", index=False)
print("✅ Saved to ➤ final_selected_features1.csv")
