import torch
import torch.nn as nn
from network import Network


class Agent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.K = kwargs["K"]
        self.F = kwargs["F"]
        self.lr1 = kwargs["lr1"]
        self.lr2 = kwargs["lr2"]
        self.lr3 = kwargs["lr3"]
        self.tau = kwargs["tau"]
        self.alpha = kwargs["alpha"]
        self.gamma = kwargs["gamma"]        

        self.loss = None
        self.Jr = None
        self.Jc = None
        self.lam = 0 

        self.net = Network(self.K, self.F).to("cuda")
        self.target_net = Network(self.K, self.F).to("cuda")
        self.huber = nn.SmoothL1Loss()

        param_score = set(self.net.score_net.parameters())
        param_rvalue = set(self.net.value_net.parameters())
        param_cvalue = set(self.net.const_net.parameters())

        self.optimizer1 = torch.optim.Adam(param_score, self.lr1)
        self.optimizer2 = torch.optim.Adam(param_rvalue | param_cvalue, self.lr2)
        self.net.load_state_dict(self.target_net.state_dict())

    def get_action(self, s, p, repre=False):
        with torch.no_grad():
            sample = self.net.sampling(s, repre)
            log_prob = self.net.log_prob(s, sample)
            sample = sample.cpu().numpy().reshape(-1)
            log_prob = log_prob.cpu().numpy()
            action = (sample - p)[1:]
        return action, sample, log_prob

    def update(self, s, p, r, c, ns, log_prob, done):
        eps_clip = 0.2
        log_prob = log_prob.view(-1, 1)
        log_prob_ = self.net.log_prob(s, p).view(-1, 1)
        ratio = torch.exp(log_prob_ - log_prob)
        entropy = self.net.entropy(s).view(-1, 1)

        # State Value Loss
        with torch.no_grad():
            next_value = self.target_net.value(ns)
            v_target = r + self.gamma * next_value * (1-done)
            v_target = v_target * torch.clamp(ratio.detach(), 1-eps_clip, 1+eps_clip)
        
        value = self.net.value(s)
        v_loss = self.huber(value, v_target)

        # Cost Value Loss
        with torch.no_grad():
            next_c_value = self.target_net.c_value(ns)
            c_target = c + self.gamma * next_c_value * (1-done)
            c_target = c_target * torch.clamp(ratio.detach(), 1-eps_clip, 1+eps_clip)

        c_value = self.net.c_value(s)
        c_loss = self.huber(c_value, c_target)

        # Actor loss
        td_advantage_r = r + self.gamma * self.net.value(ns) * (1-done) - value
        td_advantage_c = c + self.gamma * self.net.c_value(ns) * (1-done) - c_value
        td_advantage = (td_advantage_r - self.lam * td_advantage_c).detach()

        surr1 = ratio * td_advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * td_advantage
        actor_loss = -torch.min(surr1, surr2) - 0.01 * entropy
        actor_loss = actor_loss.mean()
        
        # Update
        self.Jr = value.mean().detach().cpu().item()
        critic_loss = v_loss + c_loss
        self.loss = critic_loss + actor_loss

        self.optimizer2.zero_grad()
        critic_loss.backward()
        self.optimizer2.step()

        self.optimizer1.zero_grad()
        actor_loss.backward()
        self.optimizer1.step()
        return v_loss, c_loss, actor_loss

    def update_lam(self, s):
        c_value = self.net.c_value(s)
        self.Jc = c_value.mean().detach().cpu().item()
        lam_grad = -(self.Jc - self.alpha)
        self.lam -= self.lr3 * lam_grad
        self.lam = max(self.lam, 0)

    def soft_target_update(self):
        for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def hard_target_update(self):
        self.net.load_state_dict(self.target_net.state_dict())


