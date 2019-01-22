"""
Learning purposes -- adapted from Siraj Raval's "RL for Stock Prediction"
"""

from keras.models import load_model

from agent import Agent
from functions import *
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if len(sys.argv) != 2:
    # print("Usage: python evaluate.py [stock] [model]")
    print("Usage: python evaluate.py [stock]")
    exit()

for eNum in range(10, 1001, 10):
    # stock_name, model_name = sys.argv[1], sys.argv[2]
    stock_name = sys.argv[1]
    model_name = "model_ep" + eNum
    model = load_model("../models/SR_models/" + model_name)
    window_size = model.layers[0].input.shape.as_list()[1]

    agent = Agent(window_size, True, model_name)
    data = getStockDataVec(stock_name)
    l = len(data) - 1
    batch_size = 32

    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    buy = [0] * len(data)
    sell = [0] * len(data)

    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))
            buy[t] = formatPrice(data[t])

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
            sell[t] = formatPrice(data[t])

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print(stock_name + " Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

            if total_profit > 0:
                print("Winner winner chicken dinner: " + model_name)
                graph(data, buy, sell, model_name)
