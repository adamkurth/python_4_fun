# The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
# This is just to demonstrate gradient descent.

from numpy import *
import csv

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    data = {'w': [], 'b': [], 'error': []}  # Initialize data dictionary for storing weights, biases, and errors
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        error = compute_error_for_line_given_points(b, m, points)
        data['b'].append(b)
        data['w'].append(m)
        data['error'].append(error)
    return [b, m, data]

def write_data_to_csv(data):
    with open('output_grad.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(['w', 'b', 'error'])
        for i in range(len(data['w'])):
            write.writerow([data['w'][i], data['b'][i], data['error'][i]])

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # Initial y-intercept guess
    initial_m = 0  # Initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                               compute_error_for_line_given_points(
                                                                                   initial_b, initial_m, points)))
    print("Running...")
    [b, m, data] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, data['error'][-1]))
    write_data_to_csv(data)

if __name__ == '__main__':
    run()
