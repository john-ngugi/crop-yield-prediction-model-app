import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers

st.title('Crop Yield Prediction Model')

def main():
    np.set_printoptions(precision=3, suppress=True)
    url = "https://raw.githubusercontent.com/john-ngugi/SSCM-AI-ML/main/katukefarm.csv"
    column_names = ['YEARS', 'YIELD_PRODUCTION', 'NDVI', 'SARVI', 'NPCRI', 'RVI', 'GCI']
    raw_dataset = pd.read_csv(url, delimiter=',', header=None, skiprows=1, names=column_names)

    # Display dataset
    st.subheader('Raw Dataset')
    st.write(raw_dataset)

    # Preprocessing
    dataset = raw_dataset.copy()
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('YIELD_PRODUCTION')
    test_labels = test_features.pop('YIELD_PRODUCTION')

    train_features.pop('YEARS')
    test_features.pop('YEARS')

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Define the model
    def build_and_compile_model():
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(train_features.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    dnn_model = build_and_compile_model()
    st.subheader('Model Summary')
    dnn_model.summary(print_fn=lambda x: st.text(x))

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, restore_best_weights=True)

    # Train the model
    history = dnn_model.fit(
        train_features_scaled, train_labels.values,
        validation_split=0.2,
        epochs=10000,
        verbose=0,
        callbacks=[early_stopping]
    )

    # Plot training history
    def plot_loss(history):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error [YIELD]')
        plt.legend()
        plt.grid(True)
        return plt

    st.subheader('Training Loss')
    st.pyplot(plot_loss(history))

    # Evaluate the model
    test_results = {}
    test_results['dnn_model'] = dnn_model.evaluate(test_features_scaled, test_labels.values, verbose=0)

    st.subheader('Model Performance')
    st.write(pd.DataFrame(test_results, index=['Mean Absolute Error [YIELD_PRODUCTION]']).T)

    # Predictions
    test_predictions = dnn_model.predict(test_features_scaled).flatten()

    # Plot predictions
    def plot_predictions(true, predicted):
        plt.figure(figsize=(8, 6))
        plt.scatter(true, predicted)
        plt.xlabel('True Values [YIELD]')
        plt.ylabel('Predictions [YIELD]')
        lims = [0, 10000]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims, 'r--')
        return plt

    st.subheader('Predicted vs. Actual')
    st.pyplot(plot_predictions(test_labels.values, test_predictions))

    # Display predictions and true labels
    st.subheader('Predicted Values')
    st.write(test_predictions)
    st.subheader('Actual Values')
    st.write(test_labels)

if __name__ == '__main__':
    main()
