# Future Sales Prediction using Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create Sample Monthly Sales Data
data = {
    "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Sales": [200, 220, 250, 270, 300, 320, 350, 370, 400, 420, 450, 480]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Month"]]
y = df["Sales"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict Future Sales (Next 3 Months)
future_months = np.array([[13], [14], [15]])
future_sales = model.predict(future_months)

# Print Predictions
for i, sale in enumerate(future_sales, start=13):
    print(f"Predicted Sales for Month {i}: {round(sale, 2)}")

# Plot Graph
plt.scatter(df["Month"], df["Sales"])
plt.plot(df["Month"], model.predict(X))
plt.scatter(future_months, future_sales)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction")
plt.show()
