# Journal of progress # 

## 1/15 ## 
initialized repo and begun training adapted from Siraj Raval's Q Learning for Stock Trading tutorial. 
Configured data paths and environment setup between local and server. 

## 1/16 ##
I spun up a notebook to add graphing capabilities, adapted the notebook content into src.  

## 1/18 ## 
After training a model for 24 hours, found that it failed to converge and opted to never buy or sell. 
I believe the model was waiting for an opportune time to sell to maximize benefits, but never made the move to purchase in the first place. 

My plan now is to play around with the reward function and possibly tune some of the hyperparameters as so: 

1)  Reward: 
    - If there were more that 20 consecutive buys or 50 consecutive "no action", then I gave the agent a big negative reward like -500
    - Removed the max (in the buy action) and multiplied the reward by 100.