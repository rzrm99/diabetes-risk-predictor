

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, precision_recall_curve

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Load and Clean Data
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
df = df[df["BMI"] > 0]
df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 1 if x > 0 else 0)

# 2. Feature Engineering
df["BMI_Age"] = df["BMI"] * df["Age"]
df["BP_Chol"] = df["HighBP"] + df["HighChol"]
df["PoorHealth"] = df["GenHlth"] + df["PhysHlth"] + df["MentHlth"]

# 3. Split Data
X = df.drop(columns=["Diabetes_012", "Diabetes_binary"])
y = df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. SMOTE Balancing
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

# 6. Train XGBoost Model
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    colsample_bytree=0.8,
    subsample=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42
)
model.fit(X_res, y_res, eval_set=[(X_test_scaled, y_test)], verbose=False)

# 7. Predict & Tune Threshold
y_probs = model.predict_proba(X_test_scaled)[:, 1]
thresholds = np.linspace(0.1, 0.9, 50)
f1s = [f1_score(y_test, y_probs > t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1s)]
y_pred = (y_probs > best_thresh).astype(int)

print(f"\nBest Threshold: {best_thresh:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Confusion Matrix
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, fmt='d', cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 9. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# 10. Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_probs)
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.show()

# 11. SHAP Explainability (Optional)
explainer = shap.Explainer(model)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, features=X_test, feature_names=X.columns)
