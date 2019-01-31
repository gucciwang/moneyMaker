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
    
## 1/21 ## 
Adjusted as noted above, but still no convergence. Now experimenting with action replay. Will be turning the 
sampling method to random. 

## 1/30 ## 
Adjusted the reward function in the sell action to give negative reward if the sell lost money. Hoorah! Finally an optimial model. However, the model seems to 
diverge after the 110th episode out of 1000 episodes... Not sure why. Perhaps some more hyperparameter tuning or maybe needs a wider window? 
This model was trained with 1 year of S&P 500 data (Jan 2018 -> Jan 2019) with a window size of 20 and 1000 episodes. 

![Winner](images/SR_GSPC_20_1000/model_ep30.png "Winner!")

Next steps: 
- Play around with hyperparameters 
- Try expanding the dataset of S&P 500 to a couple of years 