# Candidate-Elimination Algorithm Implementation (Without CSV)

import copy

# Training Data: [Sky, AirTemp, Humidity, Wind, Water, Forecast, EnjoySport]
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

# Initialize S and G
num_attributes = len(data[0]) - 1

S = ['Ø'] * num_attributes
G = [['?'] * num_attributes]

def is_consistent(hypothesis, example):
    for i in range(len(hypothesis)):
        if hypothesis[i] != '?' and hypothesis[i] != example[i]:
            return False
    return True

for example in data:
    if example[-1] == "Yes":  # Positive Example
        for i in range(num_attributes):
            if S[i] == 'Ø':
                S[i] = example[i]
            elif S[i] != example[i]:
                S[i] = '?'
        
        # Remove inconsistent hypotheses from G
        G = [g for g in G if is_consistent(g, example)]

    else:  # Negative Example
        new_G = []
        for g in G:
            if is_consistent(g, example):
                for i in range(num_attributes):
                    if S[i] != '?' and S[i] != example[i]:
                        new_hypothesis = copy.deepcopy(g)
                        new_hypothesis[i] = S[i]
                        new_G.append(new_hypothesis)
        G = new_G

# Remove duplicates
G_unique = []
for g in G:
    if g not in G_unique:
        G_unique.append(g)

print("Final Specific Hypothesis (S):")
print(S)

print("\nFinal General Hypothesis (G):")
for g in G_unique:
    print(g)
