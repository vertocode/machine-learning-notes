# Classification where the result should be 0 or 1 (false or true) is called binary classification.
# For binary classification, sigmoid activation function should be used.
# For output layer, the sigmoid activation function is recommended to be used in case the result can be 1 or 0.
# Dense(units=3, activation='sigmoid')

# For regression problems where the result should be a negative or positive number, we should
# use the linear activation function.
# Dense(units=1, activation='linear')

# For regression problems where the result should be non-negative, we should
# use the ReLU activation function. [Faster learning, and recommended to use in hidden layers when possible.]
# Dense(units=1, activation='relu')

# model recommended example
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
model = Sequential([
    Dense(units=25, activation="relu"), # input, and hidden layer
    Dense(units=15, activation="relu"), # hidden layer
    Dense(units=1, activation="sigmoid") # output layer
])

# Also take a look in the file multiclass_classification.py to see how to use the softmax activation function.
