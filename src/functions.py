"""
Learning purposes -- adapted from Siraj Raval's "RL for Stock Prediction"
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("../data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])

# Generates a graph upon evaluation to show when the model buys and sells
def graph(data, buy, sell, name, profit, stock_name, total_spent):
    # reformatting
    days = list(range(len(data)))
    for i in range(len(data)):
        if buy[i] == 0:
            buy[i] = float('nan')

        if sell[i] == 0:
            sell[i] = float('nan')

    # plotting
    plt.figure(figsize=(15,5))
    plt.plot(days, data, 'b',
         days, buy, 'go',
         days, sell, 'rx')
    plt.xlabel('Days')
    plt.ylabel('$$$')
    # plt.title(name + ", stock: " + stock_name + ", profit: " + str(profit))
    plt.title("{}, stock: {}, profit: {}, started with: {}".format(name, stock_name, profit, total_spent))
    plt.legend(['Price','buy','sell'])
    plt.savefig('../images/' + name + "_" + stock_name + '.png', format='png', dpi=1200)
    plt.close()
