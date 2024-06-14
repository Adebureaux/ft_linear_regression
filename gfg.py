import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.b = np.array([0.0, 0.0])  # Initialize coefficients theta0 and theta1

    def update_coeffs(self, learning_rate):
        Y_pred = self.predict()
        m = len(self.Y)
        self.b[0] -= learning_rate * (1 / m) * np.sum(Y_pred - self.Y)
        self.b[1] -= learning_rate * (1 / m) * np.sum((Y_pred - self.Y) * self.X)

    def predict(self, X=None):
        if X is None:
            X = self.X
        return self.b[0] + self.b[1] * X

    def compute_cost(self, Y_pred):
        m = len(self.Y)
        J = (1 / (2 * m)) * np.sum((Y_pred - self.Y)**2)
        return J

    def plot_best_fit(self, Y_pred, fig):
        plt.figure(fig)
        plt.scatter(self.X, self.Y, color='b')
        plt.plot(self.X, Y_pred, color='g')
        plt.show()

def main():
    X = np.array([i for i in range(11)])
    Y = np.array([2*i for i in range(11)])

    regressor = LinearRegression(X, Y)

    iterations = 0
    max_iterations = 100
    steps = 100
    learning_rate = 0.005
    costs = []

    # original best-fit line
    Y_pred = regressor.predict()

    while iterations < max_iterations:
        Y_pred = regressor.predict()
        cost = regressor.compute_cost(Y_pred)
        costs.append(cost)
        regressor.update_coeffs(learning_rate)
        iterations += 1

    # final best-fit line
    Y_pred = regressor.predict()
    regressor.plot_best_fit(Y_pred, 'Final Best Fit Line')

    # plot to verify cost function decreases
    plt.figure('Verification')
    plt.plot(range(iterations), costs, color='b')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function vs Iterations')
    plt.show()

if __name__ == '__main__':
    main()
