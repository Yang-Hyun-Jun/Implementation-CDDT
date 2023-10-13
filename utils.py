import torch
import numpy as np

def tensorize(array):
    tensor = torch.tensor(array[np.newaxis]).float().to("cuda")
    return tensor 

def make_batch(transition):
    x = list(zip(*transition))
    x = list(map(torch.cat, x))
    return x

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    data1 = pd.read_csv('Metrics/seed1/Jc', index_col=0)['Jc'].values.reshape(-1,1)
    data2 = pd.read_csv('Metrics/seed2/Jc', index_col=0)['Jc'].values.reshape(-1,1)
    data3 = pd.read_csv('Metrics/seed3/Jc', index_col=0)['Jc'].values.reshape(-1,1)
    data4 = pd.read_csv('Metrics/seed4/Jc', index_col=0)['Jc'].values.reshape(-1,1)
    data5 = pd.read_csv('Metrics/seed5/Jc', index_col=0)['Jc'].values.reshape(-1,1)

    data = np.concatenate([data1, data2, data3, data4, data5], axis=1)
    
    expec = np.mean(data, axis=1) 
    std = np.std(data, axis=1)
    beta = 1.5 
    cl1 = 'C2'


    plt.fill_between(x=np.arange(expec.shape[0]), y1=expec + beta * std, y2=expec - beta * std, alpha=0.3, color=cl1)
    plt.plot(expec, label="mean", color=cl1)
    plt.hlines(y=2.6, xmin=0, xmax=2000, colors='black')
    plt.title('Jc', fontsize=20)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    
