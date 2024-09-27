from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

# Lambda is the regularization parameter.
# If you want to increase or decrease the regularization, you can adjust the value of λ.
# Increasing it helps prevent overfitting, but too much may lead to underfitting.
# Decreasing it can reduce underfitting by allowing the model to learn more complex patterns, but it may increase the risk of overfitting.
# Be careful when adjusting this value to avoid introducing overfitting or underfitting issues.
λ = l2(0.01)

layer1 = Dense(units=25, activation="relu", kernel_regularizer=λ)
layer2 = Dense(units=15, activation="relu", kernel_regularizer=λ)
layer3 = Dense(units=1, activation="relu", kernel_regularizer=λ)

model = Sequential([layer1, layer2, layer3])

model.summary()
