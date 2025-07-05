import sys
import tensorflow as tf
import tensorflow_datasets as tfds

""" 
    A pygame tutorial and visualisation of how handwriting detection works using machine learing and training a Convolutional Neural Network to recognize
        handwriting from the EMNIST dataset.
"""

# Load EMNIST Letters dataset (1-26 -> A-Z)
(ds_letters_train, ds_letters_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Preprocess letters: Normalize, reshape, label from 10-35 (A-Z)
def preprocess_letters(image, label):
    image = tf.cast(image, tf.float32) / 255.0

    # reshape to exact shape to match digits
    image = tf.reshape(image, (28, 28, 1))
    label = label + 9
    label = tf.one_hot(label, 36)
    return image, label

# Apply map to EMNIST sets
ds_letters_train = ds_letters_train.map(preprocess_letters)
ds_letters_test = ds_letters_test.map(preprocess_letters)

# Load emnist digits dataset to repare data for training, and use in a CNN
(x_digits_train, y_digits_train), (x_digits_test, y_digits_test) = tf.keras.datasets.mnist.load_data()


# Normalize and reshape
x_digits_train, x_digits_test = x_digits_train / 255.0, x_digits_test / 255.0
x_digits_train = x_digits_train.astype('float32')
x_digits_test = x_digits_test.astype('float32')
x_digits_train = x_digits_train.reshape(-1, 28, 28, 1)
x_digits_test = x_digits_test.reshape(-1, 28, 28, 1)

# One-hot encode to 36 classes
y_digits_train = tf.cast(tf.keras.utils.to_categorical(y_digits_train, num_classes=36), tf.float32)
y_digits_test = tf.cast(tf.keras.utils.to_categorical(y_digits_test, num_classes=36), tf.float32)

# Convert to tf.data.Dataset
ds_digits_train = tf.data.Dataset.from_tensor_slices((x_digits_train, y_digits_train))
ds_digits_test = tf.data.Dataset.from_tensor_slices((x_digits_test, y_digits_test))

# Now we want to combine both data sets ie, digits + letters
ds_train = ds_digits_train.concatenate(ds_letters_train).shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_digits_test.concatenate(ds_letters_test).batch(32).prefetch(tf.data.AUTOTUNE)

# Create a convolutional neural network, instead of .add, we're using a list of layers we want
model = tf.keras.models.Sequential([

    # 1- Convolutional layer. Learn 32 filters using a 3x3 kernel, first layer, applying convolution, 32 filters, 3x3 kernel
    # we specify what the input shape is, not 4, but 28, 28, 1, we have a 2D array, with 1 channel value, ie, bw not colour(RGB) 
    tf.keras.layers.Conv2D(
        32, (6, 6), activation="relu", input_shape=(28, 28, 1)
    ),

    # 2- Max-pooling layer, using 2x2 pool size, extracting max value from 2x2 grid
    tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),

    # 3- Flatten units, into a single layer, to pass into NN, also our input layer
    tf.keras.layers.Flatten(),

    # 3.1- Add a hidden layer with dropout, with 192 units, to prevent overfitting we add a dropout, randomly drop half(cause 0.5) nodes from hidden layer
    # dense =  each node is connected with each of the nodes in the previous layer
    tf.keras.layers.Dense(192, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # 3.2 Add an output layer with 36 units (0–9 and A–Z), ie categories of classification 0-9, AND 26 letters and what the softmax activation function does,
    # is that it turns the output into a P distribution.
    tf.keras.layers.Dense(36, activation="softmax")
])

# Train and complie neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(ds_train, epochs=10)

# Evaluate neural network performance
model.evaluate(ds_test, verbose=2)

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")
