from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

# Lambda is the regularization parameter
# If you want to increase or decrease the regularization, you can change the value of λ
# It can be useful to fix overfitting and underfitting problems.
# If you increase this will solve the overfitting problem, but it can cause underfitting.
# If you decrease this will solve the underfitting problem, but it can cause overfitting.
# So be careful when you change this value to be sure that you are not causing overfitting or underfitting.
λ = l2(0.01)

layer1 = Dense(units=25, activation="relu", kernel_regularizer=λ)
layer2 = Dense(units=15, activation="relu", kernel_regularizer=λ)
layer3 = Dense(units=1, activation="relu", kernel_regularizer=λ)

model = Sequential([layer1, layer2, layer3])

model.summary()
