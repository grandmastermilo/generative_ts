import torch
from torch.utils.data import Dataset
import pandas as pd


class StocksDataset(Dataset):


    def __init__(self, path:str, seq_len:int, drop_cols:list = None) -> None:
        """
        StocksDataLoader constructor

        class for loading OHLC stock data
        """

        self.path = path
        self.seq_len = seq_len
        self.drop_cols = drop_cols

        self.data = pd.read_csv(self.path)

        if drop_cols is not None:
            self.data = self.data.drop(columns=self.drop_cols)

        return 

    
    def __len__(self):
        """
        Methods is used for returning and index to this max,
        this is the last index we can sample a sequence length from
        """
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        """
        Method is used for retreiving a batch of sequences

        returns [seq_len, num_features]

        dataloader output [batch_size, seq_len, num_features]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            raise Exception("Not implimented in the dataloader")

        sample = self.data.iloc[idx:idx+self.seq_len,:].to_numpy()
      
        return sample



if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset = StocksDataset(
        path='../../../data/processed/GOOGLE_BIG.csv',
        seq_len=3,
        drop_cols=['Adj_Close']
    )

    loader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
    )

    for batch_idx, samples in enumerate(loader):
        print(batch_idx, samples)
        input()
