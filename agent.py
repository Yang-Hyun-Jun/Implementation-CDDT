import torch
import torch.nn as nn
from network import Network


class Agent:
    def __init__(self, **kwargs):

        self.K = kwargs["K"]
        self.F = kwargs["F"]
        self.lr1 = kwargs["lr1"]
        self.lr2 = kwargs["lr2"]
        self.lr3 = kwargs["lr3"]
        self.tau = kwargs["tau"]
        self.alpha = kwargs["alpha"]
        self.gamma = kwargs["gamma"]        

        self.lam = 0.0
        self.net = Network(self.K, self.F).to("cuda")
        self.target_net = Network(self.K, self.F).to("cuda")
        self.target_net.load_state_dict(self.net.state_dict())

        self.huber = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params':self.net.score_net.parameters(), 'lr':self.lr1},
                                           {'params':self.net.value_net.parameters(), 'lr':self.lr2},
                                           {'params':self.net.const_net.parameters(), 'lr':self.lr2}])


    def get_action(self, s, p, mode=False):
        with torch.no_grad():
            sample = self.net.sampling(s, mode).squeeze(0)
            log_pi = self.net.log_prob(s, sample)
            sample = sample.cpu().numpy()
            log_pi = log_pi.cpu().numpy()
            action = (sample - p)[1:]
        return action, sample, log_pi

    def update(self, s, p, r, c, ns, log_pi, done):
        
        eps_clip = 0.2
        log_pi_ = self.net.log_prob(s, p).unsqueeze(1)
        entropy = self.net.entropy(s).unsqueeze(1)
        ratio = torch.exp(log_pi_ - log_pi)

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

        a_loss = -torch.min(surr1, surr2) - 0.01 * entropy
        a_loss = a_loss.mean()
        
        # Update
        loss = v_loss + c_loss + a_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return v_loss, c_loss, a_loss

    def update_lam(self, s):
        with torch.no_grad():
            r_value = self.net.value(s)
            c_value = self.net.c_value(s)
            Jr = r_value.mean().cpu().item()
            Jc = c_value.mean().cpu().item()

        lam_grad = -(Jc - self.alpha)
        self.lam -= self.lr3 * lam_grad
        self.lam = max(self.lam, 0)
        return Jr, Jc

    def soft_target_update(self):
        for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)


