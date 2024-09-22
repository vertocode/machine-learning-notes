import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import make_blobs

# make  dataset for examples
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

# Multiclass classification refers to classification problems where
# you can have more than just two possible output labels like just 0 or 1 (true and false).

# softmax example
softmaxModel = Sequential([
    Dense(units=25, activation="relu"), # input, and hidden layer
    Dense(units=15, activation="relu"), # hidden layer
    Dense(units=10, activation="softmax") # output layer using the multiclass to multiple binary classification
])

softmaxModel.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy())

softmaxModel.fit(X_train, y_train, epochs=10)

softmaxModel.summary()

# Or (for linear cases like pricing houses by size)

linearModel = Sequential([
    Dense(units=25, activation="relu"), # input, and hidden layer
    Dense(units=15, activation="relu"), # hidden layer
    Dense(units=10, activation="linear") # output layer using the multiclass to multiple binary classification
])

# This param "from_logits" is useful to be more accurate in the loss calculation.
linearModel.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

linearModel.fit(X_train, y_train, epochs=10)

linearModel.summary()

# Or (for complex models like self driving)

# Imagine a scenario where you should detect if there are pedestrians, cars, or trucks in the road by a image.
# In this case, you should use the sigmoid activation function in the output layer with 3 units, 1 for each class.
sigmoidModel = Sequential([
    Dense(units=25, activation="relu"), # input, and hidden layer
    Dense(units=15, activation="relu"), # hidden layer
    Dense(units=3, activation="sigmoid") # output layer using the multiclass to multiple
])

sigmoidModel.compile(loss = tf.keras.losses.BinaryCrossentropy())

sigmoidModel.summary()
