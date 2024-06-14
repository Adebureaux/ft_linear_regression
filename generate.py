import numpy as np
import pandas as pd

# Define parameters for generating synthetic data
np.random.seed(42)  # For reproducibility

# Number of data points
num_data_points = 1200

# True parameters (actual relationship)
true_intercept = 10000
true_slope = -0.1

# Generate mileage data
mileage = np.random.uniform(10000, 200000, num_data_points)

# Generate price data based on the true relationship with added noise
noise = np.random.normal(0, 2000, num_data_points)  # Gaussian noise with mean 0 and std dev 2000
price = true_intercept + true_slope * mileage + noise

# Create a DataFrame
data = pd.DataFrame({'Mileage': mileage, 'Price': price})

# Save the dataset to a CSV file
data.to_csv('car_data.csv', index=False)

print("Dataset generated and saved successfully.")
