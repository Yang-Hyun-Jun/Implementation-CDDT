import torch
import numpy as np
import pandas as pd

from collections import deque
from replaymemory import ReplayMemory
from environment import Environment
from agent import Agent

class Trainer:
    def __init__(self, **kwargs):
        
        self.K = kwargs["K"]
        self.F = kwargs["F"]
        self.cons = kwargs["cons"]
        self.data = kwargs["data"]       
        self.path = kwargs["path"] 
        self.balance = kwargs["balance"]
        self.episode = kwargs["episode"]
        self.holding = kwargs["holding"]
        self.batch_size = kwargs["batch_size"]
 
        self.env = Environment(kwargs["data"])
        self.memory = ReplayMemory(kwargs["memory_size"])
        self.agent = Agent(**kwargs)

    def save_model(self):
        save_path = self.path + "/net.pth"
        torch.save(self.agent.net.state_dict(), save_path)

    def save_actor(self):
        save_path = self.path + "/actor.pth"
        torch.save(self.agent.net.score_net.state_dict(), save_path)

    def tensorize(self, array):
        tensor = torch.tensor(array[np.newaxis]).float().to("cuda")
        return tensor   

    def make_batch(self, sampled_exps):
        x = list(zip(*sampled_exps))
        x = list(map(torch.cat, x))
        return x

    def train(self):
        scores_window = deque(maxlen=100)
        costs_window = deque(maxlen=100)

        v_loss = 0
        a_loss = 0
        c_loss = 0

        portfolio_values = []
        profitlosses = []
        Jrs = []
        Jcs = []

        for epi in range(1, self.episode+1):
            cum_r = 0 
            cum_c = 0
            steps_done = 0
            batch_data = None
            state = self.env.reset(self.balance)
            
            while True:
                is_hold = steps_done % self.holding != 0
                action, sample, log_prob = self.agent.get_action(self.tensorize(state), self.env.portfolio)
                action = np.zeros((self.K-1)) if is_hold else action
                next_state, reward, cost, done = self.env.step(action, sample)
                self.env.initial_balance = self.env.portfolio_value if not is_hold else self.env.initial_balance
                transition = [state, sample, reward, cost, next_state, log_prob, done]
                self.memory.push(list(map(self.tensorize, transition))) if not is_hold else 0
                
                cum_r += reward[0]
                cum_c += cost[0]
                state = next_state
                steps_done += 1

                if (len(self.memory) >= self.batch_size):
                    batch_data = self.memory.sample(self.batch_size)
                    batch_data = self.make_batch(batch_data)
                    v_loss, c_loss, a_loss = self.agent.update(*batch_data)
                    self.agent.soft_target_update()

                if (len(self.memory) >= self.batch_size) & done[0] & self.cons:               
                    self.agent.update_lam(batch_data[0])

                if (epi == self.episode):
                    portfolio_values.append(self.env.portfolio_value)
                    profitlosses.append(self.env.profitloss)
                    
                if (epi == self.episode) & done[0]:
                    pd.DataFrame({"Profitloss":profitlosses}).to_csv(self.path + "/Profitloss_Train")
                    pd.DataFrame({"PV":portfolio_values}).to_csv(self.path + "/Portfolio_Value_Train")
                    pd.DataFrame({"Jr":Jrs}).to_csv(self.path + "/Jr")
                    pd.DataFrame({"Jc":Jcs}).to_csv(self.path + "/Jc")
                    self.save_actor()
                    self.save_model()
                
                if (epi % 500 == 0) & done[0]:
                    self.save_actor()
                    self.save_model()

                if done[0]:
                    Jrs.append(self.agent.Jr)
                    Jcs.append(self.agent.Jc)
                    scores_window.append(cum_r)
                    costs_window.append(cum_c)
                    score_r = np.mean(scores_window)
                    score_c = np.mean(costs_window)

                    alpha = self.agent.net(self.tensorize(state).reshape(1,-1))
                    alpha = alpha.detach().cpu().numpy().reshape(-1)

                    print(f"epi:{epi}")
                    print(f"lam:{self.agent.lam}")
                    print(f"Jr:{self.agent.Jr}")
                    print(f"Jc:{self.agent.Jc}")                               
                    print(f"a_loss:{a_loss}")
                    print(f"v_loss:{v_loss}")
                    print(f"c_loss:{c_loss}")
                    print(f"cum_fee:{self.env.cum_fee}")
                    print(f"cum c:{cum_c}")
                    print(f"cum r:{cum_r}")
                    print(f"score_r:{score_r}")
                    print(f"score_c:{score_c}")
                    print(f"log prob:{log_prob}")
                    print(f"profitloss:{self.env.profitloss}")
                    # print(f"sample:{sample}")
                    print(f"portfolio:{self.env.portfolio}\n")
                    break