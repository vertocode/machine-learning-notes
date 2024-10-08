import tensorflow
from tensorflow.keras.layers import Dense, Input, Dot
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import MeanSquaredError

user_NN = Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32),
])


item_NN = Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32)
])

# create the user input and point to the base network
input_user = Input(shape=user_NN.input_shape[1])
vu = user_NN(input_user)
vu = tensorflow.keras.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = Input(shape=item_NN.input_shape[1])
vm = item_NN(input_item)
vm = tensorflow.keras.linalg.l2_normalize(vm, axis=1)

# measure the similarity between the user and the item
output = Dot(axes=1)([vu, vm])
print(output)

# specify the inputs and output of the model
model = Model([input_user, input_item], output)
