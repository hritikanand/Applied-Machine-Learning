import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset from the given file path
file_path = "/Users/hritikanand/Library/CloudStorage/OneDrive-SwinburneUniversity/Applied Machine Learning/Lab 02/insurance.csv"
df = pd.read_csv(file_path)

# One-hot encoding for categorical variables (converting categorical data into numerical form)
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Scaling numerical features to ensure they are on the same scale
scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])

# Splitting data into training (80%) and testing (20%) sets
X = df.drop(columns=['charges'])  # Independent variables
# Target variable
y = df['charges']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression using only BMI as a single feature
X_bmi_train, X_bmi_test = X_train[['bmi']], X_test[['bmi']]
lin_reg_bmi = LinearRegression()
lin_reg_bmi.fit(X_bmi_train, y_train)  # Training the model
y_pred_bmi = lin_reg_bmi.predict(X_bmi_test)  # Making predictions
# Calculating Mean Absolute Error 
mae_bmi = mean_absolute_error(y_test, y_pred_bmi)  

# Multiple Linear Regression using all available features
lin_reg_multi = LinearRegression()
lin_reg_multi.fit(X_train, y_train)  # Training the model
y_pred_multi = lin_reg_multi.predict(X_test)  
# Calculating MAE
mae_multi = mean_absolute_error(y_test, y_pred_multi)  

# Deep Neural Network (DNN) with Single Input (BMI only)
dnn_single = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=[1]),  # First hidden layer with 16 neurons
    layers.Dense(8, activation='relu'),  # Second hidden layer with 8 neurons
    layers.Dense(1)  # Output layer
])
dnn_single.compile(optimizer='adam', loss='mae')  # Compiling the model with Adam optimizer
dnn_single.fit(X_bmi_train, y_train, epochs=100, verbose=0, batch_size=16)  # Training the model
y_pred_dnn_single = dnn_single.predict(X_bmi_test).flatten()  
 # Calculating MAE
mae_dnn_single = mean_absolute_error(y_test, y_pred_dnn_single) 

# Deep Neural Network (DNN) with Multiple Inputs (all features)
dnn_multi = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[X_train.shape[1]]),  # First hidden layer with 32 neurons
    layers.Dense(16, activation='relu'),  # Second hidden layer with 16 neurons
    layers.Dense(1)  
])
dnn_multi.compile(optimizer='adam', loss='mae')  # Compiling the model
dnn_multi.fit(X_train, y_train, epochs=100, verbose=0, batch_size=16)  # Training the model
y_pred_dnn_multi = dnn_multi.predict(X_test).flatten()  
# Calculating MAE
mae_dnn_multi = mean_absolute_error(y_test, y_pred_dnn_multi)  

# Compare the performance of different models
model_performance = pd.DataFrame({
    "Model": ["Linear Regression (BMI)", "Multiple Linear Regression", "DNN (BMI)", "DNN (Multiple Inputs)"],
    "MAE": [mae_bmi, mae_multi, mae_dnn_single, mae_dnn_multi]
})
print(model_performance)  # Display the performance metrics

# Visualizing Predictions from the Best Model
best_model = min(zip(["BMI", "Multi", "DNN BMI", "DNN Multi"], [mae_bmi, mae_multi, mae_dnn_single, mae_dnn_multi]), key=lambda x: x[1])[0]
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_dnn_multi, color="red", alpha=0.5, label="DNN Multi Predictions")  # Plotting predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", linestyle="--", label="Ideal Prediction")  # Ideal line
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title(f"Best Model: {best_model} Predictions vs Actual Values")
plt.legend()
plt.show()
