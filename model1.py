
import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 


st.title('crop yeild prediction model') 


def main():
    np.set_printoptions(precision=3, suppress=True)
    url="https://raw.githubusercontent.com/john-ngugi/SSCM-AI-ML/main/katukefarm.csv"
    column_names = ['YEARS', 'YIELD_PRODUCTION','NDVI','SARVI', 'NPCRI', 'RVI', 'GCI']
    raw_dataset = pd.read_csv(url,delimiter=',', header=None, skiprows=1, names=column_names)
    st.dataframe(raw_dataset)
    dataset=raw_dataset.copy()
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
     #divide data into training and testing data 
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    # remove yeild prediction from the original dataset because that is what is being predicted
    train_labels = train_features.pop('YIELD_PRODUCTION')
    test_labels = test_features.pop('YIELD_PRODUCTION')    
    #remove years since they are not used for the prediction 
    train_features.pop("YEARS")
    test_features.pop("YEARS") 
    #create an array of the training data 
    train_features_array = np.array(train_features)
    #normalize the dataset to range at certain values to increase computation speed
    normalizer = tf.keras.layers.Normalization(input_shape=[5,], axis=None)
    normalizer.adapt(train_features_array)
    # define the model 

    def build_and_compile_model(norm):
      model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)])
      model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
      return model

    # show the model summary
    dnn_model = build_and_compile_model(normalizer)
    dnn_model.summary()        
    # we fit our linear model 
    history = dnn_model.fit(train_features.values,train_labels.values, validation_split=0.2,verbose=0, epochs=10000)
    
    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 10000])
        plt.xlabel('Epoch')
        plt.ylabel('Error [YIELD]')
        plt.legend()
        plt.grid(True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot_loss(history))

    test_results = {}
    test_results['dnn_model'] = dnn_model.evaluate(test_features.values, test_labels.values, verbose=0)
  
    pd.DataFrame(test_results, index=['Mean absolute error [YIELD_PRODUCTION]']).T
    
    test_predictions = dnn_model.predict(test_features.values).flatten()
    
    a = plt.subplots()
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels.values, test_predictions)
    plt.xlabel('True Values [YIELD]')
    plt.ylabel('Predictions [YIELD]')
    lims = [0, 10000]
    plt.xlim(lims)
    plt.ylim(lims)
    
    _ = plt.subplots()
    _ = plt.plot(lims, lims)

    #we show the predicted and the actual results 
    st.subheader('Predicted values')
    st.write(test_predictions)
    st.subheader('Actual values are')
    st.write(test_labels)
    
if __name__=='__main__':
     main()
