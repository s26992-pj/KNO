import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import sys

data = pd.read_csv("wine.data", header=None)

columns = ["Class", "Alcohol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
           "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols", "Proanthocyanins",
           "Color_Intensity", "Hue", "OD280/OD315", "Proline"]
data.columns = columns

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data.drop("Class", axis=1).values
y = data["Class"].values - 1

y = tf.keras.utils.to_categorical(y, num_classes=3)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def build_model_1():
    model = models.Sequential(name="Model_1_Simple")
    model.add(layers.Input(shape=(13,), name="Input"))
    model.add(layers.Dense(16, activation="relu", kernel_initializer="he_normal", name="Hidden1"))
    model.add(layers.Dense(3, activation="softmax", name="Output"))
    return model


def build_model_2():
    model = models.Sequential(name="Model_2_Deep")
    model.add(layers.Input(shape=(13,), name="Input"))
    model.add(layers.Dense(64, activation="relu", kernel_initializer="glorot_uniform", name="Hidden1"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation="relu", kernel_initializer="he_uniform", name="Hidden2"))
    model.add(layers.Dense(3, activation="softmax", name="Output"))
    return model


def train_and_plot(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.001, batch_size=16):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Val acc')
    plt.title(f'{model.name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title(f'{model.name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model.name} - Test accuracy: {acc:.4f}")
    return model, acc


model1, acc1 = train_and_plot(build_model_1(), X_train, y_train, X_test, y_test, epochs=100, lr=0.001, batch_size=16)
model2, acc2 = train_and_plot(build_model_2(), X_train, y_train, X_test, y_test, epochs=150, lr=0.0005, batch_size=8)

better_model = model1 if acc1 > acc2 else model2
print(f"\nLepszy model: {better_model.name}")

better_model.save("best_wine_model.keras")
import joblib

joblib.dump(scaler, "scaler.pkl")


def predict_from_cli():
    parser = argparse.ArgumentParser(description="Klasyfikacja wina - podaj 13 parametrów.")
    for i, col in enumerate(columns[1:]):
        parser.add_argument(f"--{col}", type=float, required=True)
    args = parser.parse_args()

    user_input = np.array([[getattr(args, col) for col in columns[1:]]])
    scaler = joblib.load("scaler.pkl")
    model = tf.keras.models.load_model("best_wine_model.keras")
    user_input = scaler.transform(user_input)

    pred = np.argmax(model.predict(user_input), axis=1)[0] + 1
    print(f" Przewidywana kategoria wina: {pred}")


if __name__ == "__main__" and len(sys.argv) > 1:
    predict_from_cli()
