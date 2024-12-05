import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load your dataset or use example data
# Assuming you already have input data as x and target values as y
# Example:
x = np.random.rand(100, 9)  # 9 features (same as your inputs)
y = np.random.rand(100)     # Target variable (Energy Dissipation)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train a simple linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Mean Absolute Error: {mae}")

# Streamlit app layout
st.title("RC Shear Wall Energy Dissipation Prediction")

# Material Properties Section
st.subheader("Material Properties")
f_c = st.number_input("Concrete Compressive Strength (f'c, MPa):", min_value=0.0)
f_y = st.number_input("Yield Strength of Reinforcement (fy, MPa):", min_value=0.0)

# Geometric Properties Section
st.subheader("Geometric Properties")
h = st.number_input("Height of Wall (h, mm):", min_value=0)
tw = st.number_input("Thickness of Wall (tw, mm):", min_value=0)

# Load Conditions Section
st.subheader("Load Conditions")
P = st.number_input("Applied Load (P, kN):", min_value=0.0)
M = st.number_input("Moment (M, kNm):", min_value=0.0)
V = st.number_input("Shear (V, kN):", min_value=0.0)

# Reinforcement and Strength Section
st.subheader("Reinforcement and Strength")
rho = st.number_input("Steel Reinforcement Ratio (ρ):", min_value=0.0)
mu = st.number_input("Flexural Strength Ratio (μ):", min_value=0.0)

# Create a button to trigger the prediction
if st.button('Predict'):
    # Construct the input feature array
    user_input = np.array([f_c, f_y, h, tw, P, M, V, rho, mu]).reshape(1, -1)

    # Predict the energy dissipation capacity using the trained model
    prediction = model.predict(user_input)

    # Display the prediction result
    st.subheader(f"Predicted Energy Dissipation: {prediction[0]:.2f} kN")
