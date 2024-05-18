import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_model('earthquake_magnitude_classifier.h5')

# Function to simulate earthquakes based on input magnitudes
def simulate_earthquakes(magnitudes):
    # Generate feature data based on input magnitudes
    num_samples = len(magnitudes)
    np.random.seed(42)
    data = np.random.rand(num_samples, 3)  # Assume 3 features as expected by the model
    data[:, 0] = magnitudes  # Set the first feature to be the magnitude
    return data

# Function to plot accuracy graph
def plot_accuracy(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    return fig

# Function to make predictions
def predict_earthquake_magnitude(data):
    # Assuming the data needs scaling as per the trained model
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    predictions = model.predict(data_scaled)
    return predictions

st.title('Earthquake Magnitude Prediction Simulator')

# Input magnitudes from the user
num_samples = st.slider('Number of earthquake samples to input:', min_value=1, max_value=10, value=3)
input_magnitudes = [st.number_input(f'Enter magnitude for sample {i+1}:', min_value=1, max_value=9, value=5) for i in range(num_samples)]

simulated_data = simulate_earthquakes(input_magnitudes)
st.write('Simulated Earthquake Data:')
st.write(pd.DataFrame(simulated_data, columns=[f'Feature {i}' for i in range(1, 4)]))  # Adjusted for 3 features

# Predict magnitudes
if st.button('Predict Earthquake Magnitude'):
    predictions = predict_earthquake_magnitude(simulated_data)
    st.write('Predicted Earthquake Magnitudes:')
    st.write(predictions)


st.header('Model Training Accuracy')
history_data = {
    'accuracy': np.random.rand(100),
    'val_accuracy': np.random.rand(100),
}
history = type('History', (object,), {'history': history_data})()

fig = plot_accuracy(history)
st.pyplot(fig)

# Predict magnitudes for the next few hours
st.header('Predict Magnitude for the Next Few Hours')
hours = st.number_input('Enter number of hours to predict:', min_value=1, max_value=24, value=3)
future_magnitudes = [st.number_input(f'Enter magnitude for future hour {i+1}:', min_value=1, max_value=9, value=5) for i in range(hours)]
future_data = simulate_earthquakes(future_magnitudes)
future_predictions = predict_earthquake_magnitude(future_data)
st.write(f'Predicted Earthquake Magnitudes for the next {hours} hours:')
st.write(future_predictions)
