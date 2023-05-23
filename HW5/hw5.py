from operator import matmul
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


def limit(hat_beta):
    result = -(hat_beta[0])/(hat_beta[1])

    print("Q6a: " + str(result))
    # TODO: Answer if the predection makes sense for what year Lake Mendota will no longer freeze
    print("Q6b: " + "The prediction might be a bit short as we are making a prediction based on linear decrease of the number of days. However, disregarding the outliers it looks like the number of forzen days is decreasing logarithmically")


def interpret(hat_beta):
    if (hat_beta[1] > 0):
        result = ">"
    elif (hat_beta[1] < 0):
        result = "<"
    else:
        result = "="

    print("Q5a: " + str(result))
    # TODO: Answer what the sign means for Mendota ice
    print("Q5b: " + "The sign represents the rate at which the number of frozen days are decreasing/increasing/or nor changing. Since it is < it means that over time the number of frozen days is decreasing and will reach a point where it is no longer frozen.")


def predict(hat_beta):
    prediction = hat_beta[0] + hat_beta[1]*2021

    print("Q4: " + str(prediction))


def linear_regress(x_values, y_values):
    # Initalize matricies
    x_matrix = np.empty([len(x_values), 2], dtype=np.int64)
    y_matrix = np.empty([len(y_values)], dtype=np.int64)

    # Generate x feature matricies
    for i in range(len(x_values)):
        x_matrix[i] = np.array([1, x_values[i]], dtype=np.int64)

    print("Q3a:")
    print(x_matrix)

    # Genereate y feature matricies
    for i in range(len(y_values)):
        y_matrix[i] = np.array([y_values[i]], dtype=np.int64)

    print("Q3b:")
    print(y_matrix)

    # Compute X^TX
    Z = np.matmul(np.transpose(x_matrix), x_matrix)

    print("Q3c:")
    print(Z)

    # Compute inverse X^TX
    I = np.linalg.inv(Z)

    print("Q3d:")
    print(I)

    # Compute pseudo inverse of x_matrix
    PI = np.matmul(I, np.transpose(x_matrix))

    print("Q3e:")
    print(PI)

    # Compute beta hat
    hat_beta = matmul(PI, y_matrix)

    print("Q3f:")
    print(hat_beta)

    return hat_beta


def visualize_data(filepath):
    with open(filepath) as file:
        # Skip first line
        file.readline()
        # Convert all data to int and create a list of coordinates
        data = [(int(x), int(y)) for x, y in csv.reader(file, delimiter=',')]

    x_values = []
    y_values = []

    # Separate the x and y's into their own arrays
    for coordinate in data:
        x_values.append(coordinate[0])
        y_values.append(coordinate[1])

    # Plot, label, and display
    plt.plot(x_values, y_values)
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.savefig("plot.jpg")

    return x_values, y_values


def main():
    filepath = sys.argv[1]
    x_values, y_values = visualize_data(filepath)
    hat_beta = linear_regress(x_values, y_values)
    predict(hat_beta)
    interpret(hat_beta)
    limit(hat_beta)


if __name__ == "__main__":
    main()
