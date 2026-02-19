# Mobile Price Prediction without CSV
# Using Random Forest Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create Sample Dataset
data = {
    "battery_power": [842, 1021, 563, 615, 1821, 1859, 1954, 1445, 509, 769],
    "ram": [2549, 2631, 2603, 2769, 1411, 3220, 700, 1099, 513, 3946],
    "px_height": [20, 905, 1263, 1216, 1208, 1004, 381, 512, 386, 441],
    "px_width": [756, 1988, 1716, 1786, 1212, 1654, 1018, 1149, 836, 874],
    "mobile_wt": [188, 136, 145, 131, 141, 164, 151, 142, 164, 146],
    "price_range": [1, 2, 2, 2, 1, 3, 0, 1, 0, 3]   # Target (0=Low,1=Medium,2=High,3=Very High)
}

df = pd.DataFrame(data)

# Features and Target
X = df.drop("price_range", axis=1)
y = df["price_range"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
