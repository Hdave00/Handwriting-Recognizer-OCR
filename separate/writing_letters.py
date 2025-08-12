import sys
import tensorflow as tf
import tensorflow_datasets as tfds

# Use EMNIST handwriting dataset -- NOTE -- EMNIST comes built in with TensorFlow lib, use it. We can make our own, but we will have to turn it into an
# array of pixels

# Load EMNIST Letters (labels 1-26 -> A-Z)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    """ Preprocess function takes the images/training data and orients them to the correct format to be able to draw on a flatplane with
        the correct angle of incidence.
          - In essence, because EMNIST letters are collected as mirrored images, but we dont want to draw them as mirrored, we 'reframe'
            them to be drawn naturally."""
    
    image = tf.cast(image, tf.float32) / 255.0

    # rotate while preserving channel
    image = tf.transpose(image, perm=[1, 0, 2])

    # horizontal flip
    image = tf.image.flip_left_right(image)

    # add channel
    image = tf.expand_dims(image, -1)

    # labels: 0-25
    label = label - 1
    label = tf.one_hot(label, 26)
    return image, label

ds_train = ds_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test  = ds_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Build model
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(40, (3, 3), activation="relu", padding="same")(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(192, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(26, activation="softmax")(x)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = tf.keras.Model(inputs, outputs)
model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )

model.summary()
model.fit(ds_train, epochs=10)

model.evaluate(ds_test)

if len(sys.argv) == 2:
    model.save(sys.argv[1])
    print(f"Letters model saved to {sys.argv[1]}")