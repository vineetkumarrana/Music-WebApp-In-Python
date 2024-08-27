import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Initialization
is_init = False
label = []
dictionary = {}
label_index = 0

# Load data and labels
for i in os.listdir():
    if i.endswith(".npy") and i.split(".")[0] != "labels":
        data = np.load(i)
        if not is_init:
            is_init = True 
            X = data
            y = np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)
            size = data.shape[0]
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = label_index  
        label_index += 1

# Convert labels to numeric form
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y)

# Shuffle the data
shuffled_indices = np.arange(X.shape[0])
np.random.shuffle(shuffled_indices)
X = X[shuffled_indices]
y = y[shuffled_indices]

# Define the model
input_shape = (X.shape[1],)
ip = Input(shape=input_shape)
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X, y, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
