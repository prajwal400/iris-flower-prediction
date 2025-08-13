import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Script started")

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Optional: Visualize the dataset
sns.pairplot(df, hue='species')
plt.show()

# Prepare data
X = df[iris.feature_names]  # pandas DataFrame with feature names
y = df['species']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Updated predict function: pass DataFrame with feature names to suppress warning
def predict_iris(sl, sw, pl, pw):
    features = pd.DataFrame([[sl, sw, pl, pw]], columns=iris.feature_names)
    return iris.target_names[model.predict(features)[0]]

# Example prediction
print(predict_iris(5.1, 3.5, 1.4, 0.2))

print("Script finished")

if __name__ == "__main__":
    print("\nEnter the measurements of the Iris flower:")

    sepal_length = float(input("Sepal length (cm): "))
    sepal_width = float(input("Sepal width (cm): "))
    petal_length = float(input("Petal length (cm): "))
    petal_width = float(input("Petal width (cm): "))

    species = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    print(f"\nPredicted Iris species: {species}")
