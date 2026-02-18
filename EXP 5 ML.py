import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Function to calculate Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# KNN Algorithm
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get k nearest samples
        k_indices = np.argsort(distances)[:self.k]

        # Get labels of k nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Create KNN classifier
classifier = KNN(k=3)
classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)

# Calculate Accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy)
