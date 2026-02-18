import math
from collections import Counter
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
def encode_dataset(data, features):
    encoders = {}
    
    
    for feature in features:
        unique_values = list(set(record[feature] for record in data))
        encoders[feature] = {value: idx for idx, value in enumerate(unique_values)}
    
    encoded_data = []
    
    for record in data:
        encoded_record = []
        for feature in features:
            encoded_record.append(encoders[feature][record[feature]])
        encoded_record.append(record[target])  # Append label
        encoded_data.append(encoded_record)
    
    return encoded_data, encoders
def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
def knn(train_data, test_instance, k=3):
    distances = []
    
    for record in train_data:
        features = record[:-1]
        label = record[-1]
        dist = euclidean_distance(features, test_instance)
        distances.append((dist, label))
    
   
    distances.sort(key=lambda x: x[0])
    
   
    k_nearest = distances[:k]
    
   
    labels = [label for _, label in k_nearest]
    prediction = Counter(labels).most_common(1)[0][0]
    
    return prediction
encoded_data, encoders = encode_dataset(data, features)
new_instance = {"MarketTrend": "Up", "Volume": "Low", "News": "Negative"}
encoded_instance = [
    encoders[feature][new_instance[feature]] for feature in features
]

prediction = knn(encoded_data, encoded_instance, k=3)

print("Prediction for new trading instance:", prediction)
