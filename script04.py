import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.src.layers import Identity
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
import os
import pickle

DATA_FILE = "wine.data"
COLUMN_NAMES = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
    'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
    'Color_intensity', 'Hue', 'OD280_OD315_ratio', 'Proline'
]
NORMALIZER_LAYER = None


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Błąd: Nie znaleziono pliku {filepath}")
        return None

    data = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.drop('Class', axis=1)
    y = data['Class'] - 1

    y_one_hot = to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def create_model(normalization_layer, units, dropout_rate, learning_rate, use_norm):
    model = Sequential()
    model.add(Input(shape=(13,)))
    model.add(normalization_layer)

    if use_norm:
        model.add(normalization_layer)
    else:
        model.add(Identity())

    model.add(Dense(units, activation='relu', kernel_initializer='he_uniform'))

    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(max(8, units // 2), activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_model_for_tuner(hp, use_norm):
    hp_units = hp.Int('units', min_value=32, max_value=128, step=16)
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    return create_model(NORMALIZER_LAYER, hp_units, hp_dropout, hp_lr, use_norm)

def run_analysis(use_norm):
    global NORMALIZER_LAYER

    data = load_data(DATA_FILE)
    if data is None: return
    X_train, X_test, y_train, y_test = data

    NORMALIZER_LAYER = Normalization(axis=-1)
    NORMALIZER_LAYER.adapt(np.array(X_train))

    baseline_model = create_model(NORMALIZER_LAYER, units=16, dropout_rate=0.0, learning_rate=0.001, use_norm=use_norm)

    baseline_model.fit(X_train, y_train, epochs=20, verbose=0)
    baseline_score = baseline_model.evaluate(X_test, y_test, verbose=0)
    print(f"BASELINE Accuracy: {baseline_score[1]:.4f} (Loss: {baseline_score[0]:.4f})")

    tuner = kt.RandomSearch(
        lambda hp: build_model_for_tuner(hp, use_norm),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='my_tuner_dir',
        project_name='wine_optimization',
        overwrite=True
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[stop_early], verbose=0)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Liczba neuronów: {best_hps.get('units')}")
    print(f"Dropout: {best_hps.get('dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    best_model = tuner.hypermodel.build(best_hps)

    history = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)

    final_score = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nBaseLine Model Accuracy: {baseline_score[1]:.4f}")
    print(f"\nTUNED Model Accuracy: {final_score[1]:.4f}")

    if final_score[1] >= baseline_score[1]:
        print(f"SUKCES: Poprawiono (lub wyrównano) wynik baseline! (+{final_score[1] - baseline_score[1]:.4f})")
    else:
        print("INFO: Tuner nie przebił baseline (zbiór jest bardzo mały/prosty).")

    best_model.save('wine_model_tuned.keras')


def predict_from_args():
    try:
        model = load_model('wine_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Błąd ładowania modelu lub scalera: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", type=int, choices=[0, 1], default=1, help="1=Tak, 0=Nie")

    if len(sys.argv) > 1 and "--norm" not in sys.argv:
        predict_from_args()
    else:
        args = parser.parse_args()
        run_analysis(use_norm=bool(args.norm))
