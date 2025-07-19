  #practical audit
  #Install necessary packages
  #!pip install aif360 scikit-learn pandas matplotlib seaborn

	# Import libraries
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

	# Load and preprocess COMPAS data
data = CompasDataset()
privileged_groups = [{'race': 'Caucasian'}]
unprivileged_groups = [{'race': 'African-American'}]
	
	# Train/test split
train, test = data.split([0.7], shuffle=True)
	
  # Train classifier
X = train.features
y = train.labels.ravel()
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
	
# Predict and wrap into BinaryLabelDataset
test_pred = model.predict(test.features)
test_pred_dataset = test.copy()
test_pred_dataset.labels = test_pred

# Fairness metrics
metrics = ClassificationMetric(test, test_pred_dataset,
	            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups)
	
print("Disparate Impact:", metrics.disparate_impact())
print("Equal Opportunity Difference:", metrics.equal_opportunity_difference())
print("False Positive Rate Difference:", metrics.false_positive_rate_difference())
	
	# Visualization
plt.bar(["Disparate Impact", "Equal Opportunity", "FPR Diff"],
	        [metrics.disparate_impact(),
	         metrics.equal_opportunity_difference(),
	         metrics.false_positive_rate_difference()],
	        color=["blue", "green", "red"])
plt.title("Racial Bias Metrics in COMPAS")
plt.ylabel("Metric Value")
plt.show()
