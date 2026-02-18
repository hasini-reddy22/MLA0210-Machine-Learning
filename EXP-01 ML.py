# FIND-S Algorithm Implementation

def find_s(training_data):
    # Initialize hypothesis to the most specific hypothesis
    hypothesis = None
    
    for sample in training_data:
        attributes = sample[:-1]   # All columns except last
        label = sample[-1]         # Last column is target label
        
        # Consider only positive examples
        if label.lower() == "yes":
            if hypothesis is None:
                # First positive example initializes hypothesis
                hypothesis = attributes.copy()
            else:
                # Generalize hypothesis where needed
                for i in range(len(hypothesis)):
                    if hypothesis[i] != attributes[i]:
                        hypothesis[i] = "?"
    
    return hypothesis


# Example Training Data
# Format: [Attribute1, Attribute2, ..., Target]
training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

# Run FIND-S
final_hypothesis = find_s(training_data)

print("Most Specific Hypothesis:")
print(final_hypothesis)
