# 🔌 Required Libraries
import os
import urllib.request
import aif360
import numpy as np
import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import joblib

# 📁 Step 1: Ensure COMPAS dataset is downloaded to the correct AIF360 directory
aif360_path = os.path.dirname(aif360.__file__)
target_dir = os.path.join(aif360_path, "data", "raw", "compas")
os.makedirs(target_dir, exist_ok=True)

dataset_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
destination = os.path.join(target_dir, "compas-scores-two-years.csv")
if not os.path.exists(destination):
    urllib.request.urlretrieve(dataset_url, destination)
    print("✅ COMPAS dataset downloaded successfully!")
else:
    print("ℹ️ Dataset already exists. Skipping download.")

# 📊 Step 2: Load the dataset using AIF360
data = CompasDataset()

# 🔍 Step 3: Dynamically inspect race encoding and set group definitions
race_values = np.unique(data.protected_attributes[:, data.protected_attribute_names.index('race')])
print(f"Detected race values: {race_values}")

# Determine which value corresponds to African-American vs. Caucasian
# This assumes the group with higher count is African-American (more samples)
race_counts = [(val, np.sum(data.protected_attributes[:, data.protected_attribute_names.index('race')] == val)) for val in race_values]
race_counts.sort(key=lambda x: x[1], reverse=True)
unprivileged_race = race_counts[0][0]
privileged_race = race_counts[1][0]

unprivileged_groups = [{'race': unprivileged_race}]
privileged_groups = [{'race': privileged_race}]

print(f"✅ Group Mapping:")
print(f"  Unprivileged race value: {unprivileged_race}")
print(f"  Privileged race value: {privileged_race}")

# 🧼 Step 4: Baseline fairness metrics before modeling
baseline = BinaryLabelDatasetMetric(data,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups)

print("\n📊 Baseline Fairness in Raw Data")
print(f"Mean difference: {baseline.mean_difference():.4f}")
print(f"Disparate Impact (raw): {baseline.disparate_impact():.4f}")

# 🔀 Step 5: Train/test split
train, test = data.split([0.7], shuffle=True)
print(f"\n🔍 Label Distribution")
print(f"Train labels: {dict(zip(*np.unique(train.labels.ravel(), return_counts=True)))}")
print(f"Test labels: {dict(zip(*np.unique(test.labels.ravel(), return_counts=True)))}")

# 📐 Optional: Visualize label distribution
sns.countplot(x=test.labels.ravel())
plt.title("Label Distribution in Test Set")
plt.xlabel("Recidivism Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 🧠 Step 6: Train Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(train.features, train.labels.ravel())

# 🤖 Step 7: Predict and wrap predictions
test_pred = model.predict(test.features)
test_pred_dataset = test.copy()
test_pred_dataset.labels = test_pred

# 📊 Step 8: Group-level prediction diagnostics
def inspect_subgroup_predictions(test_set, pred_set, attr_name='race'):
    idx = test_set.protected_attribute_names.index(attr_name)
    groups = np.unique(test_set.protected_attributes[:, idx])
    for group_val in groups:
        count = np.sum(test_set.protected_attributes[:, idx] == group_val)
        actual_pos = np.sum(test_set.labels[test_set.protected_attributes[:, idx] == group_val].ravel() == 1)
        predicted_pos = np.sum(pred_set.labels[test_set.protected_attributes[:, idx] == group_val].ravel() == 1)
        print(f"\n🧩 Group {group_val}:")
        print(f"    Total samples: {count}")
        print(f"    Actual positives: {actual_pos}")
        print(f"    Predicted positives: {predicted_pos}")

inspect_subgroup_predictions(test, test_pred_dataset)

# 📏 Step 9: Post-prediction fairness metrics
metrics = ClassificationMetric(test, test_pred_dataset,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups)

di = metrics.disparate_impact()
eo = metrics.equal_opportunity_difference()
fpr = metrics.false_positive_rate_difference()

print("\n📊 Fairness Audit Results (Post Prediction)")
print(f"Disparate Impact: {di:.4f}")
print(f"Equal Opportunity Difference: {eo:.4f}")
print(f"False Positive Rate Difference: {fpr:.4f}")

# 📉 Step 10: Visualize metrics
plt.bar(["Disparate Impact", "Equal Opportunity", "FPR Diff"],
        [di, eo, fpr],
        color=["blue", "green", "red"])
plt.title("Racial Bias Metrics in COMPAS (Post-Model)")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.show()

# 💾 Step 11: Save model
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "compas_model.pkl")
joblib.dump(model, model_path)
print(f"\n✅ Model saved to: {model_path}")
