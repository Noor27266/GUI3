import tkinter as tk
from tkinter import messagebox
import numpy as np
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
print(f"Mean Absolute Error: {mae}")

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
        user_input = np.array([f_c, f_y, h, tw, P, M, V, rho, mu]).reshape(1, -1)

        # Predict the energy dissipation capacity using the trained model
        prediction = model.predict(user_input)

        # Display the prediction result
        result_label.config(text=f"Predicted Energy Dissipation: {prediction[0]:.2f} kN")
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
