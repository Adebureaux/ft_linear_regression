import json
import os

def load_parameters(file_path):
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def predict_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def main():
    params_file = 'params.json'
    params = load_parameters(params_file)
    
    if params is None:
        print("Cannot perform prediction without parameters.")
        return
    
    theta0 = params.get('theta0')
    theta1 = params.get('theta1')
    
    if theta0 is None or theta1 is None:
        print("Error: Parameters theta0 and theta1 are required.")
        return
    
    try:
        mileage = float(input("Enter the mileage of the car: "))
    except ValueError:
        print("Error: Please enter a valid number for mileage.")
        return

    estimated_price = predict_price(mileage, theta0, theta1)
    print(f"The estimated price for a car with {mileage} mileage is: {estimated_price}")

if __name__ == '__main__':
    main()
