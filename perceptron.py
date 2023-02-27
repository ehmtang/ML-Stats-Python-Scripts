"""Reference Python Simplified https://www.youtube.com/watch?v=-KLnurhX-Pg"""

# A Perceptron is an algorithm for supervised learning of binary classifiers. 
# This algorithm enables neurons to learn and processes elements in the training set one at a time

x_input = [0.1, 0.5, 0.2] # ie my data
w_weights = [0.4, 0.3, 0.6] # these are assigned by the user,
bias = 0.5 # also known as the threshold

# Activation function to allocate is True or is False
# This case, uses a step function to determine the output 
def step(weighted_sum):
    if weighted_sum > bias:
        return 1
    else:
        return 0

# Calculates the product of input data and its weight
# Returns the output from the step function
# In this case, is it greater or less than the bias
def perceptron():
    weighted_sum = 0
    for x, w in zip(x_input, w_weights):
        weighted_sum += x*w
        print(weighted_sum)
    return step(weighted_sum)

output = perceptron()
print(f"output: {output}")