"""
Learning purposes -- adapted from Siraj Raval's "RL for Stock Prediction"
"""

from agent import Agent
from functions import *
import sys
import os

if len(sys.argv) != 4:
    print("Usage: python train.py [stock] [window] [episodes]")
    exit()

# GPU Config
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
punishment = -500

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []
    history = []

    for t in range(l):
        action = agent.act(state)
        history.append(action)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 0 and len(history) >= 50 and history[-50:] == [0] * 20:
            print("PUNISHED: 50 consecutive snoozes")
            reward = punishment

        elif action == 1:  # buy
            if len(history) >= 20 and history[-20:] == [1]*20:
                reward = punishment
                print("PUNISHED: 20 consecutive buys")
                
            else:
                agent.inventory.append(data[t])
                print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0) * 100
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save("../models/SR_models/model_ep" + str(e))
