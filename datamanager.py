import pmenv

class DataManager(pmenv.DataManager):
    Features_mo = ["momentum10", "momentum30", "momentum50", "momentum70"]
    Features_vo = ["vomentum10", "vomentum30", "vomentum50", "vomentum70"]
    Features = Features_mo + Features_vo + ["Price"]
    
    def __init__(self):
        super().__init__()

    def get_data(self, ticker:str, start_date=None, end_date=None):
        data = super().get_data(ticker, start_date, end_date)
        data = self.get_denoising(data)
        data = self.get_momentum(data)
        data = self.get_vomentum(data)
        data = data.dropna().loc[:,self.Features]
        return data 

    @staticmethod
    def get_momentum(data):
        data = data.replace(0, method="bfill")
        data.loc[data.index[10]:, "momentum10"] = (data["Adj Close"][10:].values / data["Adj Close"][:-10].values) 
        data.loc[data.index[30]:, "momentum30"] = (data["Adj Close"][30:].values / data["Adj Close"][:-30].values) 
        data.loc[data.index[50]:, "momentum50"] = (data["Adj Close"][50:].values / data["Adj Close"][:-50].values) 
        data.loc[data.index[70]:, "momentum70"] = (data["Adj Close"][70:].values / data["Adj Close"][:-70].values)     
        return data 

    @staticmethod
    def get_vomentum(data):
        data = data.replace(0, method="bfill")
        data.loc[data.index[10]:, "vomentum10"] = (data["Volume"][10:].values / data["Volume"][:-10].values)  
        data.loc[data.index[30]:, "vomentum30"] = (data["Volume"][30:].values / data["Volume"][:-30].values)     
        data.loc[data.index[50]:, "vomentum50"] = (data["Volume"][50:].values / data["Volume"][:-50].values)     
        data.loc[data.index[70]:, "vomentum70"] = (data["Volume"][70:].values / data["Volume"][:-70].values)    
        return data 
    
    @staticmethod   
    def get_denoising(data):
        data.iloc[:,1:] = data.iloc[:,1:].ewm(span=10).mean()
        return data
    
if __name__ == "__main__":
    tickers = ['COST', 'INCY', "REGN"]

    train_start = "2019-01-02"
    train_end = "2021-06-30"

    valid_start = "2021-07-01"
    valid_end = "2022-06-30" 

    test_start = "2022-07-01" 
    test_end = "2023-01-31"
    
    datamanager = DataManager()
    data1 = datamanager.get_data(tickers[0], start_date=train_start, end_date=train_end)
    data2 = datamanager.get_data(tickers[1], start_date=train_start, end_date=train_end)
    data3 = datamanager.get_data(tickers[2], start_date=train_start, end_date=train_end)

    train_data_tensor = datamanager.get_data_tensor(tickers, train_start, train_end)
    valid_data_tensor = datamanager.get_data_tensor(tickers, valid_start, valid_end)
    test_data_tensor = datamanager.get_data_tensor(tickers, test_start, test_end)
    full_data_tensor = datamanager.get_data_tensor(tickers, valid_start, test_end)

    print(data1)
    print(data2)
    print(data3)
    print(train_data_tensor.shape)
    print(valid_data_tensor.shape)
    print(test_data_tensor.shape)
    print(full_data_tensor.shape)