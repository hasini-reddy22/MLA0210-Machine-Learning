from collections import Counter, defaultdict
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
def train_naive_bayes(data, features, target):
    total_records = len(data)
    
    
    label_counts = Counter(record[target] for record in data)
    priors = {label: count / total_records for label, count in label_counts.items()}
    
    
    likelihood_counts = {
        label: {feature: Counter() for feature in features}
        for label in label_counts
    }
    
    for record in data:
        label = record[target]
        for feature in features:
            likelihood_counts[label][feature][record[feature]] += 1
    
    return priors, likelihood_counts, label_counts
def predict_naive_bayes(new_instance, priors, likelihood_counts, label_counts, features):
    total_records = sum(label_counts.values())
    posteriors = {}
    
    for label in priors:
        posterior = priors[label]
        print(f"\nCalculating for class: {label}")
        print(f"Prior P({label}) = {priors[label]:.4f}")
        
        for feature in features:
            value = new_instance[feature]
            
            
            feature_count = likelihood_counts[label][feature][value]
            total_label_count = label_counts[label]
            unique_values = len(likelihood_counts[label][feature])
            
            likelihood = (feature_count + 1) / (total_label_count + unique_values)
            
            print(f"P({feature}={value} | {label}) = {likelihood:.4f}")
            
            posterior *= likelihood
        
        posteriors[label] = posterior
        print(f"Posterior P({label}|X) = {posterior:.6f}")
    
    
    predicted_class = max(posteriors, key=posteriors.get)
    return predicted_class, posteriors
priors, likelihood_counts, label_counts = train_naive_bayes(data, features, target)
new_market_input = {
    "MarketTrend": "Up",
    "Volume": "Low",
    "News": "Negative"
}

prediction, probabilities = predict_naive_bayes(
    new_market_input,
    priors,
    likelihood_counts,
    label_counts,
    features
)

print("\nFinal Prediction:", prediction)
