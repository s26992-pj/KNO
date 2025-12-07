import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from PIL import Image
import argparse

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def save_first_test_image():
    (_, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    img = x_test[10]
    im = Image.fromarray(img)
    im.save("input.png")


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = 1.0 - (x_train.astype("float32") / 255.0)
    x_test = 1.0 - (x_test.astype("float32") / 255.0)
    return x_train, y_train, x_test, y_test


def build_fc(hp=None):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    units1 = hp.Int("units1", 64, 512, step=64) if hp else 128
    model.add(layers.Dense(units1, activation="relu"))
    dropout = hp.Float("dropout", 0.0, 0.5, step=0.1) if hp else 0.3
    model.add(layers.Dropout(dropout))
    if hp and hp.Boolean("use_dense2", default=True):
        units2 = hp.Int("units2", 32, 256, step=32)
        model.add(layers.Dense(units2, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    lr = hp.Choice("lr", [1e-2, 1e-3, 1e-4]) if hp else 1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_cnn(hp=None):
    model = models.Sequential()
    model.add(layers.Reshape((28, 28, 1), input_shape=(28, 28)))
    filters = hp.Choice("conv_filters", [16, 32, 64]) if hp else 32
    kernel = hp.Choice("kernel_size", [3, 5]) if hp else 3
    model.add(layers.Conv2D(filters, kernel, activation="relu", padding="same"))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters * 2, kernel, activation="relu", padding="same"))
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    dense_units = hp.Int("dense_units", 64, 512, step=64) if hp else 128
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    lr = hp.Choice("lr", [1e-2, 1e-3, 1e-4]) if hp else 1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_and_save(arch):
    x_train, y_train, x_test, y_test = load_data()
    model = build_cnn() if arch == "cnn" else build_fc()
    history = model.fit(x_train, y_train,
                        epochs=30, batch_size=32,
                        validation_split=0.1, verbose=1)
    model.save("model.keras")
    preds = model.predict(x_test)
    loss = history.history["loss"][-1]
    cm = confusion_matrix(y_test, np.argmax(preds, axis=1))
    np.savetxt("loss.txt", [loss])
    np.savetxt("confusion_matrix.txt", cm, fmt="%d")


def prepare_image(path):
    img = Image.open(path).convert("L").resize((28, 28))
    img = np.array(img).astype("float32") / 255.0
    img = 1.0 - img
    return img.reshape(1, 28, 28)


def predict_images(n):
    model = tf.keras.models.load_model("model.keras")
    (_, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = 1.0 - (x_test.astype("float32") / 255.0)
    img_arr = x_test[n].reshape(1, 28, 28)
    pred = model.predict(img_arr)[0]
    idx = np.argmax(pred)
    print(f"Obraz {n}: {classes[idx]} ({float(pred[idx]):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["cnn", "fc"], default="cnn")
    args = parser.parse_args()

    save_first_test_image()

    if not os.path.exists("model.keras"):
        train_and_save(arch=args.arch)

    print("Predykcje na kilku obrazkach testowych (negatywy):")
    predict_images(n=10)
