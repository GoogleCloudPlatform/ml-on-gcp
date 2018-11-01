from getdata import x_train, y_train # This is added to the exported .py file

import keras

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

optimizer = keras.optimizers.RMSprop()
loss = keras.losses.categorical_crossentropy

model.compile(optimizer, loss)

model.fit(x_train, y_train, epochs=2)
