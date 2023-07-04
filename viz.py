import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def show(seed_from=None, seed_to=None, benchmark=None, size=(5,5)):
    mean_datas = []
    mode_datas = []
    BH_datas = []
    
    seed_to = seed_from if seed_to is None else seed_to
    seed_from = 1 if seed_from is None else seed_from

    for i in range(seed_from, seed_to+1):
        path1 = os.getcwd() + f"/Metrics/seed{i}/Portfolio_Value_Test_mean"
        path2 = os.getcwd() + f"/Metrics/seed{i}/Portfolio_Value_Test_mode"
        path3 = os.getcwd() + f"/Metrics/seed{i}/Portfolio_Value_Test_BH"

        data1 = pd.read_csv(path1, index_col=0)["PV"].to_numpy()
        data2 = pd.read_csv(path2, index_col=0)["PV"].to_numpy()
        data3 = pd.read_csv(path3, index_col=0)["PV"].to_numpy()

        mean_datas.append(data1.reshape(-1,1))
        mode_datas.append(data2.reshape(-1,1))
        BH_datas.append(data3.reshape(-1,1))

    expect_mean = np.mean(np.concatenate(mean_datas, axis=-1), axis=-1)
    expect_mode = np.mean(np.concatenate(mode_datas, axis=-1), axis=-1)
    expect_BH = np.mean(np.concatenate(BH_datas, axis=-1), axis=-1)

    std_mean = np.std(np.concatenate(mean_datas, axis=-1), axis=-1)
    std_mode = np.std(np.concatenate(mode_datas, axis=-1), axis=-1)
    std_BH = np.std(np.concatenate(BH_datas, axis=-1), axis=-1)

    beta = 0.5
    cl1 = "C3"
    cl2 = "C0"
    cl3 = "C2"

    plt.figure(figsize=size)
    plt.fill_between(x=np.arange(expect_mean.shape[0]), y1=expect_mean + beta * std_mean, y2=expect_mean - beta * std_mean, alpha=0.3, color=cl1)
    plt.plot(expect_mean, label="mean", color=cl1)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)

    plt.fill_between(x=np.arange(expect_mode.shape[0]), y1=expect_mode + beta * std_mode, y2=expect_mode - beta * std_mode, alpha=0.3, color=cl2)
    plt.plot(expect_mode, label="mode", color=cl2)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)

    plt.fill_between(x=np.arange(expect_BH.shape[0]), y1=expect_BH + beta * std_BH, y2=expect_BH - beta * std_BH, alpha=1, color=cl3)
    plt.plot(expect_BH, label="BH", color=cl3)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True, color="gray", alpha=0.5)
    plt.legend(fontsize=18)
    plt.ylabel("Portfolio Value", fontsize=20)
    plt.xlabel("Test timesteps", fontsize=20)

    if benchmark is not None:
        plt.plot(benchmark, label="benchmark", color="C4")
        plt.legend(fontsize=18)
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)

    print("mean Portfolio Value:", expect_mean[-1])
    print("mode Portfolio Value:", expect_mode[-1])
    print("BH Portfolio Value:", expect_BH[-1])

    plt.show()



