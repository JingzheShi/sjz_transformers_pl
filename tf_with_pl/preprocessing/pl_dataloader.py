import pytorch_lightning as pl
import torch
import os
class PL_DataLoader(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    #def setup(self, stage=None):
    #    path=os.getcwd()
    #    self.train_data_loader=torch.load(os.path.join(path,'processed_dataset/train_data_loader.pt'))
    #    self.val_data_loader=torch.load(os.path.join(path,'processed_dataset/valid_data_loader.pt'))
        
    def naive_setup(self, train_data_loader, val_data_loader):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
    def train_dataloader(self):
        return self.train_data_loader
    def val_dataloader(self):
        return self.val_data_loader
