import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import argv


def load(path: str) -> pd.DataFrame:
    """ Load a csv, print its dimensions and return it as a DataFrame. """
    try:
        df = pd.read_csv(path)
        print(f"Loading dataset {path} of dimensions {df.shape}")
        return df
    except Exception as e:
        print(e)
    return pd.DataFrame()

def ft_abs(x):
    """ Returns absolute value of given number. """
    if x < 0:
        x *= -1
    return x

def mean_abs_error(predict, actual):
    """ Returns the mean of the loss between two lists of values. """
    loss = 0
    for pv, av in zip(predict, actual):
        loss += ft_abs(pv - av)
    loss /= len(predict)
    return loss

def get_scaled_mileage(mileage):
    """ Returns a scaled list of the mileage using standardization method. """
    mean = np.mean(mileage)
    std = np.std(mileage)
    scaled_mileage = (mileage - mean) / std
    return scaled_mileage

def update_thetas(t0, t1, learning_rate, pr_prices, act_prices, mileage):
    """ Returns the new values for theta0 and theta1 """
    m = len(pr_prices)
    prices_diff = []
    for pr, act in zip(pr_prices, act_prices):
        prices_diff.append(pr - act)
    prices_times_mileage = []
    for price, mile in zip(prices_diff, mileage):
        prices_times_mileage.append(price * mile)
    new_t0 = t0 - learning_rate * (1 / m * sum(prices_diff))
    new_t1 = t1 - learning_rate * (1 / m * sum(prices_times_mileage))
    return new_t0, new_t1

def main():
    """ Loads data from dataset file and makes multiples predictions to determine the most coherent values for theta0 and theta1. """
    if len(argv) != 2:
        print("Error: invalid number of arguments (requires path of dataset file)")
        return
    data = load(argv[1])
    if data.empty:
        return
    mileage = data["km"].to_list()
    scaled_mileage = get_scaled_mileage(mileage)
    actual_prices = data["price"].tolist()
    t0 = 0
    t1 = 0
    old_loss = 10000
    loss = 1000

    # Updates t0 and t1 until the loss converges
    while True:
        old_loss = loss
        predicted_prices = []
        for km in scaled_mileage:
            predicted_prices.append(t0 + t1 * km)
        loss = mean_abs_error(predicted_prices, actual_prices)
        if ft_abs(old_loss - loss) > 0.01:
            t0, t1 = update_thetas(t0, t1, 0.07, predicted_prices, actual_prices, scaled_mileage)
        else:
            break

    # Rescale t0 and t1 to get their actual values
    t0 = t0 - (t1 * np.mean(mileage) / np.std(mileage))
    t1 = t1 / np.std(mileage)
    print(f"Found thetas: {t0}, {t1}")

    # Writes t0 and t1 values in a file named tvalues.txt
    try:
        file = open("theta_values.txt", "w")
        file.write(str(t0) + "\n")
        file.write(str(t1) + "\n")
    except Exception as e:
        print(e)

    # Creates and show plot to compare actual and predicted values
    x_line = np.linspace(min(mileage), max(mileage), 100)
    y_line = t0 + t1 * x_line
    plt.scatter(mileage, actual_prices, color="blue", label="Actual prices")
    plt.plot(x_line, y_line, color="red", label="Predicted prices over mileage")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car prices over mileage")
    plt.legend()
    plt.grid(True, linestyle="--", alpha = 0.6)
    plt.show()

main()
