import numpy as np
data = [
    {"MarketTrend": "Up", "Volume": "High", "News": "Positive", "StockMove": "Up"},
    {"MarketTrend": "Up", "Volume": "Low", "News": "Positive", "StockMove": "Up"},
    {"MarketTrend": "Down", "Volume": "High", "News": "Negative", "StockMove": "Down"},
    {"MarketTrend": "Down", "Volume": "Low", "News": "Negative", "StockMove": "Down"},
    {"MarketTrend": "Up", "Volume": "High", "News": "Negative", "StockMove": "Up"},
    {"MarketTrend": "Down", "Volume": "High", "News": "Positive", "StockMove": "Down"},
]
features = ["MarketTrend", "Volume", "News"]
target = "StockMove"
def encode_data(data, features, target):
    encoders = {}
    
    
    for feature in features:
        unique_vals = list(set(record[feature] for record in data))
        encoders[feature] = {val: idx for idx, val in enumerate(unique_vals)}

    target_encoder = {"Down": 0, "Up": 1}
    
    X = []
    y = []
    
    for record in data:
        X.append([encoders[feature][record[feature]] for feature in features])
        y.append(target_encoder[record[target]])
    
    return np.array(X), np.array(y), encoders, target_encoder
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def train_logistic_regression(X, y, learning_rate=0.1, epochs=1000):
    n_samples, n_features = X.shape
    
    
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(epochs):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)
        
        
        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)
        
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias
def predict_probability(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    return sigmoid(linear_model)


def predict(X, weights, bias):
    probs = predict_probability(X, weights, bias)
    return [1 if p >= 0.5 else 0 for p in probs]
X, y, encoders, target_encoder = encode_data(data, features, target)

weights, bias = train_logistic_regression(X, y)

print("Trained Weights:", weights)
print("Trained Bias:", bias)
new_scenario = {
    "MarketTrend": "Up",
    "Volume": "Low",
    "News": "Negative"
}
new_X = np.array([[encoders[feature][new_scenario[feature]] for feature in features]])

probability = predict_probability(new_X, weights, bias)[0]
prediction = "Up" if probability >= 0.5 else "Down"

print("\nProbability of Stock Moving Up:", round(probability, 4))
print("Predicted Movement:", prediction)
