import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Create synthetic nonlinear dataset
np.random.seed(1)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# -------- Linear Regression --------
lin_model = LinearRegression()
lin_model.fit(X, y)
y_lin_pred = lin_model.predict(X)

lin_mse = mean_squared_error(y, y_lin_pred)
lin_r2 = r2_score(y, y_lin_pred)

# -------- Polynomial Regression (degree=2) --------
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

poly_mse = mean_squared_error(y, y_poly_pred)
poly_r2 = r2_score(y, y_poly_pred)

# Print comparison
print("Linear Regression:")
print("MSE:", lin_mse)
print("R2 Score:", lin_r2)

print("\nPolynomial Regression:")
print("MSE:", poly_mse)
print("R2 Score:", poly_r2)

# Plot results
plt.scatter(X, y, color='blue')
plt.plot(X, y_lin_pred, color='red', label='Linear')
plt.plot(X, y_poly_pred, color='green', label='Polynomial')
plt.legend()
plt.show()
