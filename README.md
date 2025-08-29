# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.


## Program:

```
/*
Program to implement the linear regression using gradient descent.
Developed by: Arun S
RegisterNumber:  212224230023
*/
```

```
/*
Program to implement the linear regression using gradient descent.
import numpy as np
import matplotlib.pyplot as plt

# Sample data: population in 10,000s and profit in $10,000s
# (X = population, Y = profit)
X = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598])
Y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233])

# Normalize data if needed (not necessary here, but helps in general)
m = len(X)  # number of training examples

# Add a column of ones to X for the bias (intercept term)
X_b = np.c_[np.ones(m), X]  # shape (m, 2)

# Initialize parameters (theta0 = intercept, theta1 = slope)
theta = np.zeros(2)  # [theta0, theta1]

# Hyperparameters
alpha = 0.01      # Learning rate
epochs = 1500     # Number of iterations

# Cost function: Mean Squared Error
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X.dot(theta)
    errors = predictions - Y
    cost = (1 / (2 * m)) * np.dot(errors, errors)
    return cost

# Gradient Descent
def gradient_descent(X, Y, theta, alpha, epochs):
    m = len(Y)
    cost_history = []

    for i in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - Y
        gradients = (1 / m) * X.T.dot(errors)
        theta -= alpha * gradients

        cost = compute_cost(X, Y, theta)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Epoch {i}: Cost = {cost:.4f}")
    
    return theta, cost_history

# Train the model
theta, cost_history = gradient_descent(X_b, Y, theta, alpha, epochs)

print(f"\nFinal parameters: theta0 = {theta[0]:.4f}, theta1 = {theta[1]:.4f}")

# Prediction function
def predict(x):
    return theta[0] + theta[1] * x

# Predict profit for a city with population 7.5 (i.e., 75,000)
pop_input = 7.5
predicted_profit = predict(pop_input)
print(f"Predicted profit for a city with population {pop_input*10000:.0f}: ${predicted_profit*10000:.2f}")

# Plotting the regression line
plt.scatter(X, Y, color='blue', label='Training data')
plt.plot(X, X_b.dot(theta), color='red', label='Linear regression')
plt.xlabel("Population (10,000s)")
plt.ylabel("Profit ($10,000s)")
plt.title("Profit Prediction using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

# Optional: Plotting the cost function over iterations
plt.plot(range(epochs), cost_history, color='purple')
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost Reduction over Time")
plt.grid(True)
plt.show()

```

## Output:

<img width="888" height="834" alt="image" src="https://github.com/user-attachments/assets/19ae784a-f134-4559-bef9-a3fe37cd5895" />

<img width="774" height="532" alt="image" src="https://github.com/user-attachments/assets/21ea408a-2af2-49f1-83f7-dff6b2b39c79" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
