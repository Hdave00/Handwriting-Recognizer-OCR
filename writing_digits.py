import sys
import tensorflow as tf
import numpy as np

# Use MNIST handwriting dataset -- NOTE -- MNIST comes built in with TensorFlow lib, use it. We can make our own, but we will have to turn it into an
# array of pixels
mnist = tf.keras.datasets.mnist

# Prepare data for training, and use in a CNN
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# taking data and dividing by 255
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

# Build model
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = tf.keras.Model(inputs, outputs)
model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )

model.summary()
model.fit(x_train, y_train, epochs=10,)

model.evaluate(x_test, y_test, verbose=2)

if len(sys.argv) == 2:
    model.save(sys.argv[1])
    print(f"Digits model saved to {sys.argv[1]}")