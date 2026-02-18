import math
from collections import Counter
def entropy(data, target_attr):
    label_counts = Counter([record[target_attr] for record in data])
    total = len(data)
    
    ent = 0.0
    for count in label_counts.values():
        probability = count / total
        ent -= probability * math.log2(probability)
    
    return ent
def information_gain(data, attr, target_attr):
    total_entropy = entropy(data, target_attr)
    
    
    values = set(record[attr] for record in data)
    
    weighted_entropy = 0.0
    total = len(data)
    
    for value in values:
        subset = [record for record in data if record[attr] == value]
        probability = len(subset) / total
        weighted_entropy += probability * entropy(subset, target_attr)
    
    return total_entropy - weighted_entropy

def id3(data, attributes, target_attr):
    labels = [record[target_attr] for record in data]
    
 
    if len(set(labels)) == 1:
        return labels[0]
    
    if not attributes:
        return Counter(labels).most_common(1)[0][0]
    
   
    gains = {attr: information_gain(data, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    
    tree = {best_attr: {}}
    
    values = set(record[best_attr] for record in data)
    
    for value in values:
        subset = [record for record in data if record[best_attr] == value]
        
        if not subset:
            tree[best_attr][value] = Counter(labels).most_common(1)[0][0]
        else:
            remaining_attrs = [attr for attr in attributes if attr != best_attr]
            subtree = id3(subset, remaining_attrs, target_attr)
            tree[best_attr][value] = subtree
    
    return tree
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    
    root_attr = next(iter(tree))
    value = sample.get(root_attr)
    
    subtree = tree[root_attr].get(value)
    
    if subtree is None:
        return None  # Unknown case
    
    return predict(subtree, sample)

data = [
    {"MarketTrend": "Up", "Volume": "High", "News": "Positive", "StockMove": "Up"},
    {"MarketTrend": "Up", "Volume": "Low", "News": "Positive", "StockMove": "Up"},
    {"MarketTrend": "Down", "Volume": "High", "News": "Negative", "StockMove": "Down"},
    {"MarketTrend": "Down", "Volume": "Low", "News": "Negative", "StockMove": "Down"},
    {"MarketTrend": "Up", "Volume": "High", "News": "Negative", "StockMove": "Up"},
    {"MarketTrend": "Down", "Volume": "High", "News": "Positive", "StockMove": "Down"},
]

attributes = ["MarketTrend", "Volume", "News"]
target = "StockMove"

tree = id3(data, attributes, target)

print("Decision Tree:")
print(tree)

new_record = {"MarketTrend": "Up", "Volume": "Low", "News": "Negative"}

prediction = predict(tree, new_record)
print("\nPrediction for new record:", prediction)
