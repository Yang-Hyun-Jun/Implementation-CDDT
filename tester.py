import torch
import numpy as np
import pandas as pd 
from trainer import Trainer
from network import Network

class Tester(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test(self, model, tests=["BH", "mean", "mode"]):
        for test in tests:
            self.agent.net = model
            self.agent.net.eval()
            state = self.env.reset(self.balance)
            cum_r = 0
            cum_c = 0 
            steps_done = 0  

            portfolio_values = []   
            profitlosses = []
            portfolios = []
            actions = []       

            while True:
                is_hold = steps_done % self.holding != 0
                action, sample, _ = self.agent.get_action(self.tensorize(state), self.env.portfolio, test)
                action = np.zeros((self.K-1)) if is_hold else action
                next_state, reward, cost, done = self.env.step(action, sample)
                
                cum_r += reward
                cum_c += cost
                state = next_state
                steps_done += 1

                portfolio_values.append(self.env.portfolio_value)
                profitlosses.append(self.env.profitloss)
                portfolios.append(self.env.portfolio)
                actions.append(action)
                
                if done:
                    pd.DataFrame({"Profitloss":profitlosses}).to_csv(self.path + f"/Profitloss_Test_{test}")
                    pd.DataFrame({"PV":portfolio_values}).to_csv(self.path + f"/Portfolio_Value_Test_{test}")
                    pd.DataFrame(portfolios).to_csv(self.path + f"/Portfolios_Test_{test}")
                    pd.DataFrame(actions).to_csv(self.path + f"/Actions_Test_{test}")
                    break
