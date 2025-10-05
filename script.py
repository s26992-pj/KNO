import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import json

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model_path = "model.keras"

# sprawdzenie czy model istnieje
if os.path.exists(model_path):
    print("Model already exist")
    model = keras.models.load_model(model_path)
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
    with open("history.json", "w") as f:
        json.dump(history.history, f)

    model.save(model_path)

history = None
if os.path.exists("history.json"):
    with open("history.json", "r") as f:
        hist_data = json.load(f)

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = input("Podaj ścieżkę do obrazu")

img = tf.keras.utils.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, 0)

# Predykcja
predictions = model.predict(img_array)
predicted = tf.argmax(predictions[0]).numpy()
print(f"Przewidywana cyfra: {predicted}")

plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.show()

# Krzywa uczenia
if hist_data:
    plt.figure(figsize=(12, 5))

    # Strata
    plt.subplot(1, 2, 1)
    plt.plot(hist_data['val_loss'], color='red', linestyle='--', label='Strata (walidacja)')
    plt.plot(hist_data['loss'], color='blue', label='Strata (trening)')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.title('Krzywa Straty')
    plt.legend()

    # Dokładność
    plt.subplot(1, 2, 2)
    plt.plot(hist_data['val_accuracy'], color='green', linestyle='--', label='Dokładność (walidacja)')
    plt.plot(hist_data['accuracy'], color='orange', label='Dokładność (trening)')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.title('Krzywa Dokładności')
    plt.legend()

    plt.show()
