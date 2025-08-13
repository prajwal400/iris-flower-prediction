from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

# Load data and train model
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model
joblib.dump(model, "iris_model.joblib")
print("Model saved!")
