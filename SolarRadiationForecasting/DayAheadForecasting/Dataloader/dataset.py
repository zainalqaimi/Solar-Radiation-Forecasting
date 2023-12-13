from torch.utils.data import Dataset, DataLoader

class TiDEDataset(Dataset):
    def __init__(self, Xw, Xt, Xh, Yl, Yh):
        self.Xw = Xw
        self.Xt = Xt
        self.Xh = Xh
        self.Yl = Yl
        self.Yh = Yh

    def __len__(self):
        return len(self.Yh)  # assuming all inputs and output are same length

    def __getitem__(self, idx):
        # Will need to transpose X at some point
        return (self.Xw[idx], self.Xt[idx], self.Yl[idx], self.Xh[idx]), self.Yh[idx]



