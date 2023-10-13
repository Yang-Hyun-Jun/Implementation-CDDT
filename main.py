import pandas as pd
import numpy as np
import wandb
import argparse
import torch
import viz 

from env import Environment
from memory import ReplayMemory
from agent import Agent
from utils import tensorize, make_batch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--lr1", type=float, default=1e-6)
parser.add_argument("--lr2", type=float, default=1e-6)
parser.add_argument("--lr3", type=float, default=5e-4)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--alpha", type=float, default=2.5)
parser.add_argument("--episode", type=float, default=2000)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--batch_size", type=float, default=128)
parser.add_argument("--memory_size", type=float, default=100000)
parser.add_argument("--balance", type=float, default=14560.05)
parser.add_argument("--holding", type=float, default=5)
parser.add_argument("--cons", type=bool, default=True)
args = parser.parse_args()

train_data = np.load('Data/train_data_tensor_10.npy')
test_data = np.load('Data/test_data_tensor_10.npy')

K = train_data.shape[1]
F = train_data.shape[2]

parameters= {
            "lr1":args.lr1, 
            "lr2":args.lr2, 
            "lr3":args.lr3, 
            "tau":args.tau, 
            "alpha":args.alpha,
            "gamma":args.gamma,
            "K":K, "F":F, 
            }

if __name__ == '__main__':

    # Train Loop
    env = Environment(train_data)
    memory = ReplayMemory(args.memory_size)
    agent = Agent(**parameters)

    PVs = []
    PFs = []
    Jrs = []
    Jcs = []

    for epi in range(1, args.episode+1):
        Jr, Jc = 0, 0
        steps, cumr, cumc = 0, 0, 0
        a_loss, v_loss, c_loss, entropy = None, None, None, None
        state = env.reset(args.balance)

        while True:
            is_hold = steps % args.holding != 0

            if is_hold:
                action = np.zeros((K-1)) if is_hold else action
                next_state, _, _, done = env.step(action)
                state = next_state
                steps += 1

            else:
                action, sample, log_pi = agent.get_action(tensorize(state), env.portfolio)
                next_state, reward, cost, done = env.step(action, sample)
                env.initial_balance = env.portfolio_value
                transition = [state, sample, reward, cost, next_state, log_pi, done]
                memory.push(list(map(tensorize, transition)))

                cumr += reward[0]
                cumc += cost[0]
                state = next_state
                steps += 1

            if (len(memory) >= args.batch_size):
                batch = memory.sample(args.batch_size)
                batch = make_batch(batch)
                v_loss, c_loss, a_loss, entropy = agent.update(*batch)
                agent.soft_target_update()
                
                if done[0] & args.cons:
                    Jr, Jc = agent.update_lam(batch[0])
                    Jrs.append(Jr)
                    Jcs.append(Jc)

            if (epi == args.episode):
                PVs.append(env.portfolio_value)
                PFs.append(env.profitloss)

            if (epi == args.episode) & done[0]:
                pd.DataFrame({'Jr':Jrs}).to_csv(f'Metrics/seed{args.seed}/Jr')
                pd.DataFrame({'Jc':Jcs}).to_csv(f'Metrics/seed{args.seed}/Jc')

            if done[0]:
                print(f"epi:{epi}")
                print(f"lam:{agent.lam}")
                print(f"Jr:{Jr}")
                print(f"Jc:{Jc}")                               
                print(f"a_loss:{a_loss}")
                print(f"v_loss:{v_loss}")
                print(f"c_loss:{c_loss}")
                print(f"cumc:{cumc}")
                print(f"cumr:{cumr}")
                print(f"log pi:{log_pi}")
                print(f"pf:{env.profitloss}\n")
                break


    # Test Loop
    env = Environment(test_data)
    agent.net.eval()
    
    mode = 'mode'
    steps = 0
    PVs = []
    PFs = []
    POs = []
    Cs = []

    state = env.reset(args.balance)

    while True:
        is_hold = steps % args.holding != 0
        
        if is_hold: 
            action = np.zeros((K-1))
            next_state, _, _, done = env.step(action)
            state = next_state
            steps += 1

        else:
            action, sample, _ = agent.get_action(tensorize(state), env.portfolio, mode)
            next_state, reward, cost, done = env.step(action, sample)
            env.initial_balance = env.portfolio_value
            state = next_state
            steps += 1

        PVs.append(env.portfolio_value)
        PFs.append(env.profitloss)
        POs.append(env.portfolio)
        Cs.append(cost)

        if done[0]:
            pd.DataFrame({'Profitloss':PFs}).to_csv(f'Metrics/seed{args.seed}/Profitloss_Test_{mode}')
            pd.DataFrame({'PV':PVs}).to_csv(f'Metrics/seed{args.seed}/Portfolio_Value_Test_{mode}')
            pd.DataFrame({'C':Cs}).to_csv(f'Metrics/seed{args.seed}/Cost_Test_{mode}')
            pd.DataFrame(POs).to_csv(f'Metrics/seed{args.seed}/Portfolios_Test_{mode}')
            break

viz.show(args.seed, size=(20,8))