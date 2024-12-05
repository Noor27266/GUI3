import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model

# Load the dataset (make sure you replace this with your actual dataset)
# Example:
# data = pd.read_csv("your_data.csv")
# Ensure the data is formatted correctly for training
# For this example, we will create dummy data:
data = pd.DataFrame({
    'f_c': np.random.rand(100),  # Concrete Compressive Strength
    'f_y': np.random.rand(100),  # Yield Strength of Reinforcement
    'h': np.random.rand(100),    # Height of Wall
    'tw': np.random.rand(100),   # Thickness of Wall
    'P': np.random.rand(100),    # Applied Load
    'M': np.random.rand(100),    # Moment
    'V': np.random.rand(100),    # Shear
    'rho': np.random.rand(100),  # Steel Reinforcement Ratio
    'mu': np.random.rand(100),   # Flexural Strength Ratio
    'Energy': np.random.rand(100)  # Target (Energy Dissipation)
})

# Separate input features (X) and target variable (Y)
Input = data.iloc[:, :-1].values
Target = data.iloc[:, -1].values

# Transpose to match input/output shape if necessary
x = Input.T
t = Target.T

# Split the data into training, validation, and test sets
train_size = int(0.8 * x.shape[1])
val_size = int(0.1 * x.shape[1])
x_train, x_val, x_test = x[:, :train_size], x[:, train_size:train_size+val_size], x[:, train_size+val_size:]
t_train, t_val, t_test = t[:train_size], t[train_size:train_size+val_size], t[train_size+val_size:]

# Define the ANN model
model = models.Sequential()
model.add(layers.Input(shape=(x.shape[0],)))  # Input layer
model.add(layers.Dense(10, activation='relu'))  # Hidden layer
model.add(layers.Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['accuracy'])

# Train the model
model.fit(x_train.T, t_train.T, epochs=200, batch_size=10, validation_data=(x_val.T, t_val.T))

# Predictions
y_train = model.predict(x_train.T)
y_test = model.predict(x_test.T)
y_val = model.predict(x_val.T)

# Performance Metrics
MAE = mean_absolute_error(t_test, y_test)
R2 = r2_score(t_test, y_test)

print(f"MAE: {MAE}")
print(f"R2 Score: {R2}")

# Save the model
model.save('F:/Graphical User Interface/GUI3/ann_model.h5')
print("Model training complete and saved as ann_model.h5.")

# Function to make predictions on new data
def predict_input():
    try:
        # Get user input values from the entry fields
        f_c = float(entry_f_c.get())  # Concrete Compressive Strength (f'c)
        f_y = float(entry_f_y.get())  # Yield Strength of Reinforcement (fy)
        h = float(entry_h.get())      # Height of Wall (h)
        tw = float(entry_tw.get())    # Thickness of Wall (tw)
        P = float(entry_P.get())      # Applied Load (P)
        M = float(entry_M.get())      # Moment (M)
        V = float(entry_V.get())      # Shear (V)
        rho = float(entry_rho.get())  # Steel Reinforcement Ratio (ρ)
        mu = float(entry_mu.get())    # Flexural Strength Ratio (μ)

        # Construct the input feature array
        user_input = np.array([f_c, f_y, h, tw, P, M, V, rho, mu])

        # Load the trained model
        model = load_model('F:/Graphical User Interface/GUI3/ann_model.h5')

        # Predict the energy dissipation capacity using the trained model
        prediction = model.predict(user_input.reshape(1, -1))

        # Display the prediction result
        result_label.config(text=f"Predicted Energy Dissipation: {prediction[0][0]:.2f} kN")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for all fields.")

# Create main window
root = tk.Tk()
root.title("RC Shear Wall Energy Dissipation Prediction")

# Material Properties Section
frame_material = tk.LabelFrame(root, text="Material Properties", padx=10, pady=10)
frame_material.grid(row=0, column=0, padx=10, pady=5, sticky="w")

tk.Label(frame_material, text="Concrete Compressive Strength (f'c, MPa):").grid(row=0, column=0, sticky="w")
entry_f_c = tk.Entry(frame_material)
entry_f_c.grid(row=0, column=1)

tk.Label(frame_material, text="Yield Strength of Reinforcement (fy, MPa):").grid(row=1, column=0, sticky="w")
entry_f_y = tk.Entry(frame_material)
entry_f_y.grid(row=1, column=1)

# Geometric Properties Section
frame_geometry = tk.LabelFrame(root, text="Geometric Properties", padx=10, pady=10)
frame_geometry.grid(row=1, column=0, padx=10, pady=5, sticky="w")

tk.Label(frame_geometry, text="Height of Wall (h, mm):").grid(row=0, column=0, sticky="w")
entry_h = tk.Entry(frame_geometry)
entry_h.grid(row=0, column=1)

tk.Label(frame_geometry, text="Thickness of Wall (tw, mm):").grid(row=1, column=0, sticky="w")
entry_tw = tk.Entry(frame_geometry)
entry_tw.grid(row=1, column=1)

# Load Conditions Section
frame_loads = tk.LabelFrame(root, text="Load Conditions", padx=10, pady=10)
frame_loads.grid(row=2, column=0, padx=10, pady=5, sticky="w")

tk.Label(frame_loads, text="Applied Load (P, kN):").grid(row=0, column=0, sticky="w")
entry_P = tk.Entry(frame_loads)
entry_P.grid(row=0, column=1)

tk.Label(frame_loads, text="Moment (M, kNm):").grid(row=1, column=0, sticky="w")
entry_M = tk.Entry(frame_loads)
entry_M.grid(row=1, column=1)

tk.Label(frame_loads, text="Shear (V, kN):").grid(row=2, column=0, sticky="w")
entry_V = tk.Entry(frame_loads)
entry_V.grid(row=2, column=1)

# Reinforcement and Strength Section
frame_reinforcement = tk.LabelFrame(root, text="Reinforcement and Strength", padx=10, pady=10)
frame_reinforcement.grid(row=3, column=0, padx=10, pady=5, sticky="w")

tk.Label(frame_reinforcement, text="Steel Reinforcement Ratio (ρ):").grid(row=0, column=0, sticky="w")
entry_rho = tk.Entry(frame_reinforcement)
entry_rho.grid(row=0, column=1)

tk.Label(frame_reinforcement, text="Flexural Strength Ratio (μ):").grid(row=1, column=0, sticky="w")
entry_mu = tk.Entry(frame_reinforcement)
entry_mu.grid(row=1, column=1)

# Prediction Button
predict_button = tk.Button(root, text="Predict", command=predict_input)
predict_button.grid(row=4, column=0, pady=10)

# Result Display Section
result_label = tk.Label(root, text="Predicted Energy Dissipation: N/A", font=("Arial", 14))
result_label.grid(row=5, column=0, pady=10)

# Run the application
root.mainloop()
