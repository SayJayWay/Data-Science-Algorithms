# Simple neural network
import numpy as np
import math
import random

# Creating a perceptron.  "Fires" if weighted sum of inputs is >= 0
def step_function:
    return 1 if x>= 0 else 0

def perceptron_output(weights,bias,x):
    """returns 1 if the perceptron 'fires', 0 if not"""
    calculation = np.dot(weights,x) + bias
    return step_function(calculation)

""" With these two functions we can create simple logic gates
AND gate: weights = [2,2]; bias = -3;

OR gate: weights = [2,2]; bias = -1;

NOT gate: weights = [-2]; bias = 1;
"""

# Typically we would have discrete layers of neurons that are connected
def sigmoid(t):
    return 1/(1+math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(np.dot(weights, inputs))

# Now we can represent our neural network as a list of (noninput) layers, where
# each layer is just a list of the neurons in that layer (i.e. a list (layers)
# of lists (neurons) of lists (weights)

# Our feed-forward neural network
def feed_forward(neural_network, input_vector):
    """ takes in a neural network (list of lists of lists of weights)
        and returns the output from forward-propagating the input
        This will produce output of every neuron
        feed_foward[-1] is the outputs of the output-layer neurons"""

    outputs=[]

    # process one layer at a time
    for layer in neural_network:
        input_with_bias = input_vector + [1]                # add a bias input
        output = [neuron_output(neuron, input_with_bias)    # compute the output
                  for neuron in layer]                      # for each neuron
        outputs.append(output)                              # and remember it

        # then the input to the next layer is the output of current one
        input_vector = output
    return outputs

""" Example: XOR gate
xor_network = [# hidden layer
                [[20,20,-30],    # 'and' neuron
                 [20,20,-10]],   # 'or' neuron
               # output layer
               [[-60,60,-30]]]  # '2nd input but not 1st input' neuron

To implement:
    for x in [0,1]:
        for y in [0,1]:
            print(x,y,feed_forward(xor_network,[x,y])[-1]
"""

# For larger problems, we have to train our neural networks as we won't know
# what our neurons should be.  We can use backpropagation to do so
def backpropagate(network, input_vector, targets):
    hidden_outputs = feed_forward(network, input_vector)

    # the output * (1-output) is from the derivative of the sigmoid
    output_deltas = [output*(1-output)*(output-target)
                     for output, target in zip(outputs, targets)]

    # adjust weights for output layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # adjust the jth weight based on both
            # this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i]*hidden_output

    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output*(1-hidden_output)*
                     np.dot(output_deltas,[n[i] for n in output_layer])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i]*input

""" Example: CAPTCHA (determining digits from 0-9), where the numbers will be
a vector of length 25 whose elements are either 1 (in the image) or 0 ('this
pixel is not in the image')

for instance, the digit 0 would be:

zero = [1,1,1,1,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        1,1,1,1,1]

We will also need 10 outputs for each corresponding number.  For example,
detection of the number 0 may output something like:

output = [0,1,0,0,0,0,0,0,0,0]

targets = [[1 if i==j else 0 for i in range(10)]
            for j in range(10)]

input_size = 25
num_hidden = 5
output_size = 10

# each hidden neuron has one weight per input, plus a bias weight
hidden_layer = [[random.random() for __ in range(num_hidden + 1)]
                for __ in range(num_hidden)]

# each output has one weight per hidden neuron, plus a bias weight
output_layer = [[random.random() for __ in range(num_hidden+1)]
                for __ in range(output_size)]

# network starts with random weights
network = [hidden_layer, output_layer]

Now we can train our algorithm with backpropagation:
for __ in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)

And testing our model on the training set:
def predict(input):
    return feed_forward(network, input)[-1]
