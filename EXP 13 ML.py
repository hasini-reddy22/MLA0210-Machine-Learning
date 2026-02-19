import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Create Sample Car Dataset
# -----------------------------
data = {
    'Engine_Size': [1.2, 1.5, 1.8, 2.0, 2.2, 1.3, 1.6, 2.5, 3.0, 2.8],
    'Mileage': [20, 18, 15, 12, 10, 22, 17, 8, 6, 7],
    'Age': [5, 4, 6, 3, 2, 7, 5, 1, 1, 2],
    'Price': [500000, 600000, 750000, 900000, 1100000,
              480000, 650000, 1500000, 2000000, 1800000]
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Split Features & Target
# -----------------------------
X = df[['Engine_Size', 'Mileage', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# -----------------------------
# Step 3: Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Predict Test Data
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -----------------------------
# Step 6: Predict New Car Price
# -----------------------------
new_car = pd.DataFrame(
    [[2.0, 14, 3]],
    columns=['Engine_Size', 'Mileage', 'Age']
)

predicted_price = model.predict(new_car)

print("\nPredicted Car Price: ₹", int(predicted_price[0]))
