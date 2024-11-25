import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from concurrent.futures import ThreadPoolExecutor

st.title('Crop Yield Prediction Model')


def load_and_prepare_data(url, column_names):
    raw_dataset = pd.read_csv(url, delimiter=',', header=None, skiprows=1, names=column_names)
    dataset = raw_dataset.copy()
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop('YIELD_PRODUCTION')
    test_labels = test_features.pop('YIELD_PRODUCTION')
    train_features.pop("YEARS")
    test_features.pop("YEARS")
    return train_features, test_features, train_labels, test_labels


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10000])
    plt.xlabel('Epoch')
    plt.ylabel('Error [YIELD]')
    plt.legend()
    plt.grid(True)
    return plt


def main():
    np.set_printoptions(precision=3, suppress=True)
    url = "https://raw.githubusercontent.com/john-ngugi/SSCM-AI-ML/main/katukefarm.csv"
    column_names = ['YEARS', 'YIELD_PRODUCTION', 'NDVI', 'SARVI', 'NPCRI', 'RVI', 'GCI']

    with ThreadPoolExecutor() as executor:
        future_data = executor.submit(load_and_prepare_data, url, column_names)
        train_features, test_features, train_labels, test_labels = future_data.result()

    st.subheader('Raw Dataset')
    st.write(pd.read_csv(url, delimiter=',', header=None, skiprows=1, names=column_names))

    train_features_array = np.array(train_features)
    normalizer = tf.keras.layers.Normalization(input_shape=[5, ], axis=None)
    normalizer.adapt(train_features_array)

    dnn_model = build_and_compile_model(normalizer)
    st.subheader('Model Summary')
    dnn_model.summary(print_fn=lambda x: st.text(x))

    history = dnn_model.fit(train_features.values, train_labels.values, validation_split=0.2, verbose=0, epochs=100)

    with ThreadPoolExecutor() as executor:
        future_plot = executor.submit(plot_loss, history)
        plt_fig = future_plot.result()

    st.pyplot(plt_fig)

    test_results = {}
    test_results['dnn_model'] = dnn_model.evaluate(test_features.values, test_labels.values, verbose=0)

    test_predictions = dnn_model.predict(test_features.values).flatten()

    fig, ax = plt.subplots()
    ax.scatter(test_labels.values, test_predictions)
    ax.set_xlabel('True Values [YIELD]')
    ax.set_ylabel('Predictions [YIELD]')
    ax.set_xlim([0, 10000])
    ax.set_ylim([0, 10000])
    ax.plot([0, 10000], [0, 10000], 'r--')
    st.pyplot(fig)

    st.subheader('Predicted Values')
    st.write(test_predictions)
    st.subheader('Actual Values')
    st.write(test_labels)


if __name__ == '__main__':
    main()
