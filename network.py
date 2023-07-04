import torch
import torch.nn as nn
import numpy as np

from torch.distributions.dirichlet import Dirichlet

class Network(nn.Module):
    def __init__(self, K, F):
        super().__init__()

        self.K = K
        self.F = F

        self.score_net = nn.Sequential(
            nn.Linear(F * K, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, K),
            )

        self.value_net = nn.Sequential(
            nn.Linear(F * K , 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),            
            )

        self.const_net = nn.Sequential(
            nn.Linear(F * K, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            )

    def forward(self, x):
        """
        state: [Batch Num, K, 3]
        3: 종목 당 가격 변화율, 종목 당 포트폴리오 비율, 현재 쿠션
        """
        return self.score_net(x)
    
    def entropy(self, state):
        """
        Dirichlet Dist의 현재 entropy
        """
        alpha = self.alpha(state)
        dirichlet = Dirichlet(alpha)
        return dirichlet.entropy()
    
    def c_value(self, state):
        """
        Expected Sum of Cost
        """
        state = state.reshape(-1, self.F * self.K)
        c = self.const_net(state)
        return c

    def value(self, state):
        """
        Critic의 Value
        """
        state = state.reshape(-1, self.F * self.K)
        v = self.value_net(state)
        return v

    def alpha(self, state):
        """
        Dirichlet Dist의 Concentration Parameter
        """
        state = state.reshape(-1, self.F * self.K)
        scores = self(state).reshape(-1, self.K)
        scores = torch.clamp(scores, -40., 500.)
        alpha = torch.exp(scores) + 1
        return alpha

    def log_prob(self, state, portfolio):
        """
        Dirichlet Dist에서 샘플의 log_prob
        """
        alpha = self.alpha(state)
        dirichlet = Dirichlet(alpha)
        return dirichlet.log_prob(portfolio)

    def sampling(self, state, mode=False):
        """
        Dirichlet Dist에서 포트폴리오 샘플링
        """
        alpha = self.alpha(state).detach()
        dirichlet = Dirichlet(alpha)
        sampled_p = None 
        
        B = alpha.shape[0]  
        N = alpha.shape[1]  

        if mode == "mean":
            sampled_p = dirichlet.mean

        elif mode == "mode":
            sampled_p = dirichlet.mode

        elif mode == "BH":
            sampled_p = torch.ones(size=(N,)) / N
            sampled_p = sampled_p.to("cuda")

        elif not mode:
            sampled_p = dirichlet.sample([1])[0]
        
        return sampled_p