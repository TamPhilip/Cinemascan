from random import random
import math
from numpy import array
from numpy import cumsum

def sigmoid(x):
    1 / (1 + math.exp(-1 * x))

# math.tanh()

def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y

X, y = get_sequence(10)
print(X)
print(X.shape)
print(y)
print(y.shape)