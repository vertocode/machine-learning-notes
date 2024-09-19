import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from data import load_coffee_data
from neurons_utils import plt_roast

def coffeeSequentialModel(x, y, test_data):
    # Normalized the data
    print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

    # Tile/copy our data to increase the training set size and reduce the number of training epochs.
    Xt = np.tile(Xn,(1000,1))
    Yt= np.tile(Y,(1000,1))

    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential([
        tf.keras.Input(shape=(2,)),
        Dense(units=3, activation='sigmoid'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    )
    model.fit(Xt, Yt, epochs=10)

    X_testn = norm_l(test_data)
    predictions = model.predict(X_testn)

    model.summary()

    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            print(f"Prediction: {predictions[i][0]:0.2f} - Good Coffee")
        else:
            print(f"Prediction: {predictions[i][0]:0.2f} - Bad Coffee")


X, Y = load_coffee_data()

test_data = np.array([
    [200, 13.9], # good coffee
    [200, 17] # bad coffee
])

coffeeSequentialModel(X, Y, test_data)

# Show the data
plt_roast(X, Y)
