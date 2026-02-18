import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights randomly
np.random.seed(1)

input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

# Weight matrices
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))

wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Learning rate
lr = 0.1

# Training the network
for epoch in range(10000):

    # Forward Propagation
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wout) + bout
    predicted_output = sigmoid(final_input)

    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(wout.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Update weights and biases
    wout += hidden_output.T.dot(d_predicted_output) * lr
    bout += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden_layer) * lr
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

# Final Output
print("Final Predicted Output after Training:")
print(predicted_output)
