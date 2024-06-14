import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data.csv')  # Assuming the dataset is in a file named 'data.csv'
mileage = data['km'].values
price = data['price'].values

# Normalize the dataset
mileage_mean = np.mean(mileage)
mileage_std = np.std(mileage)
mileage_normalized = (mileage - mileage_mean) / mileage_std

price_mean = np.mean(price)
price_std = np.std(price)
price_normalized = (price - price_mean) / price_std

# Number of training examples
m = len(mileage)

# Initialize theta0 and theta1
theta0, theta1 = 0.0, 0.0

# Hyperparameters
learning_rate = 0.01  # Adjust as necessary
num_iterations = 1000

# Gradient Descent
for _ in range(num_iterations):
    # Compute the predicted prices
    predictions = theta0 + theta1 * mileage_normalized
    
    # Compute the errors
    errors = predictions - price_normalized
    
    # Update theta0 and theta1
    temp_theta0 = theta0 - learning_rate * (1 / m) * np.sum(errors)
    temp_theta1 = theta1 - learning_rate * (1 / m) * np.sum(errors * mileage_normalized)

    # Update theta values
    theta0, theta1 = temp_theta0, temp_theta1

    # Print progress every 100 iterations
    if _ % 100 == 0:
        print(f"Iteration {_}: theta0 = {theta0}, theta1 = {theta1}")

# Denormalize theta1 to get the actual slope
theta1_actual = theta1 * (price_std / mileage_std)
theta0_actual = price_mean - theta1_actual * mileage_mean

# Save the parameters as Python scalars
params = {'theta0': theta0_actual, 'theta1': theta1_actual}
with open('params.json', 'w') as f:
    json.dump(params, f)

print(f"Training completed. Parameters saved: theta0 = {theta0_actual}, theta1 = {theta1_actual}")

# Predictions using the final model
final_predictions = theta0_actual + theta1_actual * mileage

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.absolute((final_predictions - price) / price)) * 100

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plotting the results
plt.scatter(mileage, price, color='blue', label='Data points')
plt.plot(mileage, final_predictions, color='red', label='Regression line')
plt.text(0.05, 0.95, f'MAPE: {mape:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Mileage vs Price')
plt.legend()
plt.show()
