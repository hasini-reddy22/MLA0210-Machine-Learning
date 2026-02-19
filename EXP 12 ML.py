import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Step 1: Load Iris Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

# -----------------------------
# Step 2: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# -----------------------------
# Step 3: Create KNN Model
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# -----------------------------
# Step 4: Predict Test Data
# -----------------------------
y_pred = knn.predict(X_test)

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)

# -----------------------------
# Step 6: Predict New Flower
# -----------------------------
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(new_flower)

print("\nPredicted Flower Class:",
      iris.target_names[prediction][0])
