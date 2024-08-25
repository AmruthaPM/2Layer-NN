import numpy as np

# sigmoid function -> transforms a value to a value in between 0 and 1.
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X=np.array([  [0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]  ])

# output dataset
Y=np.array([ [0,0,1,1] ]).T

# seeding random values for calculation.
np.random.seed(1)

# initialize the weights randomly with mean 0
syn0=2*np.random.random( (3,1) )-1

# training phase
for i in range(10000):

    # forward propagation
    layer0=X
    layer1=nonlin(np.dot(layer0,syn0))

    # calculating error
    layer1_error=Y-layer1

    # product of error with the slope of the sigmoid at the values in layer0
    layer1_delta=layer1_error * nonlin(layer1,True)

    # updating weights
    syn0 += np.dot(layer0.T, layer1_delta)

print("OUTPUT:", layer1)

