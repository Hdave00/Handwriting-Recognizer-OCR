import sys
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.utils.class_weight import compute_class_weight # for weighting in the future

""" 
    A pygame tutorial and visualisation of how handwriting detection works using machine learning and training a Convolutional Neural Network to recognize
        handwriting from the EMNIST dataset.

        - We also do the testing for the most recently trained CNN here.
"""

# Load both datasets, ie, digits and letters
(digits_train, digits_test), digits_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Ltters data set
(letters_train, letters_test), letters_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def preprocess_digits(image, label):

    image = tf.cast(image, tf.float32) / 255.0

    # add channel
    image = tf.expand_dims(image, -1)

    # Digits are 0-9, we'll map them to 0-9 in combined labels
    label = tf.one_hot(label, 36)  # 10 digits + 26 letters
    return image, label


def preprocess_letters(image, label):

    image = tf.cast(image, tf.float32) / 255.0

    # rotate
    image = tf.transpose(image, perm=[1, 0, 2])

    # flip
    image = tf.image.flip_left_right(image)

    # add channel
    image = tf.expand_dims(image, -1)

    # letters are 1-26 in EMNIST, so map to 10-35 in combined labels
    label = label + 9  # shift to make room for digits 0-9
    label = tf.one_hot(label, 36)  # 10 digits + 26 letters
    return image, label

# Combine and shuffle datasets, because we want to be able to detect both numbers and letters
def combine_datasets(ds1, ds2):

    ds1 = ds1.map(preprocess_digits)
    ds2 = ds2.map(preprocess_letters)
    return ds1.concatenate(ds2)


# Train dataset, batch it for block work and prefetch
train_dataset = combine_datasets(digits_train, letters_train).shuffle(10000).batch(32)
test_dataset = combine_datasets(digits_test, letters_test).batch(32)


# Build the CNN using the Keras functional API... for now
inputs = tf.keras.Input(shape=(28, 28, 1))

# Careful with the adjustments, it took 13 iterations to get the right one that doesn't overfit and maintains accuracy
x = tf.keras.layers.Conv2D(40, (3, 3), activation="relu", padding="same")(inputs)

# max pooling size of 2x2, we dont want to increase else we might be iverfitting
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

# flatten layers to get the correct output format we want 
x = tf.keras.layers.Flatten()(x)

# 192 layers in the neural network using rectified linear unit ie, f(x) = max(0, x) meaning it outputs the input directly if it's positive, and 0 otherwise 
x = tf.keras.layers.Dense(192, activation="relu")(x)

# we want to randomly drop 0.4 or ~40% of the nodes in the network, to prevent overfitting to ultimately have better generalisation
x = tf.keras.layers.Dropout(0.4)(x)

# outputs 36 classes for numbers and letters ie, 0-9 = 10 digits and A-Z = 26 letters
outputs = tf.keras.layers.Dense(36, activation="softmax")(x)  

model = tf.keras.Model(inputs, outputs)

# Compile & train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", "precision", "recall"],
)

# Training it for 10 epochs/iterations, each time learning by inference as well, using the test set for inference
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Augemnent data to randomize/scramble the data even more to avoid bias
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# print the functional graph
model.summary()

# Evaluate and maybe save
if len(sys.argv) == 2:
    model.save(sys.argv[1])
    print(f"Model saved to {sys.argv[1]}")
