import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pathlib
import numpy as np

DATA_DIR = "data/images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 50
LATENT_DIM = 2

data_dir = pathlib.Path(DATA_DIR)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels=None,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

train_ds = train_ds.map(lambda x: x / 255.0)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

encoder_inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(encoder_inputs)

x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)

shape_before_flatten = x.shape[1:]

x = layers.Flatten()(x)
latent = layers.Dense(LATENT_DIM, name="latent_vector")(x)

encoder = keras.Model(encoder_inputs, latent, name="encoder")
encoder.summary()

decoder_inputs = keras.Input(shape=(LATENT_DIM,))

x = layers.Dense(int(np.prod(shape_before_flatten)))(decoder_inputs)
x = layers.Reshape(shape_before_flatten)(x)

x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

decoder_outputs = layers.Conv2D(
    3, 3, padding="same", activation="sigmoid"
)(x)

decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
decoder.summary()

autoencoder_outputs = decoder(encoder(encoder_inputs))

autoencoder = keras.Model(
    encoder_inputs,
    autoencoder_outputs,
    name="autoencoder"
)

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)

autoencoder.summary()

autoencoder.fit(
    train_ds,
    epochs=EPOCHS
)

def show_reconstructions(model, dataset, n=5):
    for batch in dataset.take(1):
        originals = batch[:n]
        reconstructions = model.predict(originals)

        plt.figure(figsize=(n * 2, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(originals[i])
            plt.axis("off")
            if i == 0:
                ax.set_title("Oryginał")

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructions[i])
            plt.axis("off")
            if i == 0:
                ax.set_title("Rekonstrukcja")

        plt.tight_layout()
        plt.show()

show_reconstructions(autoencoder, train_ds)

n = 10
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)

plt.figure(figsize=(8, 8))
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z = np.array([[xi, yi]])
        decoded = decoder.predict(z)
        ax = plt.subplot(n, n, i * n + j + 1)
        plt.imshow(decoded[0])
        plt.axis("off")

plt.suptitle("Generowanie obrazów z przestrzeni latentnej (2D)")
plt.tight_layout()
plt.show()
