import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save("model.keras")
model = keras.models.load_model("model.keras")

image_path = input("Podaj ścieżkę do obrazu (28x28 lub większy, grayscale): ")

img = tf.keras.utils.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, 0)  # dodanie wymiaru batch

predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions[0]).numpy()
print(f"Przewidywana cyfra: {predicted_class}")

plt.imshow(img, cmap='gray')
plt.title(f"Przewidywana cyfra: {predicted_class}")
plt.show()