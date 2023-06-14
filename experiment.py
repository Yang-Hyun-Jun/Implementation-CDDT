import argparse
import torch
import utils
import viz 
import os

from network import Network
from datamanager import DataManager
from trainer import Trainer
from tester import Tester

utils.nasdaq100.remove('FISV')
parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+", default=utils.nasdaq100[:90])
parser.add_argument("--train_start", type=str, default="2021-01-01")
parser.add_argument("--train_end", type=str, default="2022-03-20") 
parser.add_argument("--test_start", type=str, default="2022-03-20")
parser.add_argument("--test_end", type=str, default="2023-02-28")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--lr1", type=float, default=1e-5)
parser.add_argument("--lr2", type=float, default=5e-5)
parser.add_argument("--lr3", type=float, default=1e-4)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--alpha", type=float, default=2.2)
parser.add_argument("--episode", type=float, default=2000)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--batch_size", type=float, default=512)
parser.add_argument("--memory_size", type=float, default=100000)
parser.add_argument("--balance", type=float, default=14560.05)
parser.add_argument("--holding", type=float, default=5)
parser.add_argument("--cons", type=bool, default=False)
args = parser.parse_args()

path = os.getcwd() + f"/Metrics/seed{args.seed}"

datamanager = DataManager()
train_data_tensor = datamanager.get_data_tensor(args.tickers, args.train_start, args.train_end)
test_data_tensor = datamanager.get_data_tensor(args.tickers, args.test_start, args.test_end)

K = train_data_tensor.shape[1]
F = train_data_tensor.shape[2]-1

parameters= {
            "lr1":args.lr1, 
            "lr2":args.lr2, 
            "lr3":args.lr3, 
            "tau":args.tau, 
            "alpha":args.alpha,
            "gamma":args.gamma,
            "K":K, "F":F, 
            "alpha":args.alpha,
            "cons":args.cons,
            "balance":args.balance, 
            "holding":args.holding,
            "episode":args.episode,
            "batch_size":args.batch_size,
            "memory_size":args.memory_size,
            "path":path
            }


trainer = Trainer(**parameters, data=train_data_tensor)
trainer.train() 

test_model = Network(K, F).to("cuda")
test_model.score_net.load_state_dict(torch.load(path + "/actor.pth"))
test_model.eval()

tester = Tester(**parameters, data=test_data_tensor)
tester.test(test_model)

viz.show(args.seed, size=(20,8))