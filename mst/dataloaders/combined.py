import torch
from mst.dataloaders.cambridge import CambridgeDataset
from mst.dataloaders.medley import MedleyDBDataset
import pytorch_lightning as pl



class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
        #return self.data[i]

    def __len__(self):
        #min(len(d) for d in self.datasets)
        return  min(len(d) for d in self.datasets)
    

class CombinedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset_1: torch.utils.data.Dataset,
        train_dataset_2: torch.utils.data.Dataset,
        length: int = 524288,
        num_workers: int = 4,
        batch_size: int = 16,
        min_tracks: int = 4,
        max_tracks: int = 20,
        train_buffer_size_gb: float = 2.0,
        val_buffer_size_gb: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # self.num_workers = num_workers
        # self.batch_size = batch_size
        self.train_dataset_1 = train_dataset_1
        self.train_dataset_2 = train_dataset_2
        
    
    def setup(self, stage=None):

        if stage == "fit":
            self.train_dataset_1 = self.train_dataset_1(root_dirs = self.hparams.root_dirs,
                                                        indices = [0,150],
                                                        min_tracks = self.hparams.min_tracks,
                                                        max_tracks = self.hparams.max_tracks,
                                                        length = self.hparams.length,
                                                        buffer_size_gb=self.hparams.train_buffer_size_gb,
                                                        num_examples_per_epoch=10000,
                                                        )
            self.train_dataset_2 = self.train_dataset_2(root_dirs = self.hparams.root_dirs, 
                                                        subset = "train",
                                                        min_tracks = self.hparams.min_tracks, 
                                                        max_tracks = self.hparams.max_tracks, 
                                                        length = self.hparams.length, 
                                                        buffer_size_gb=self.hparams.train_buffer_size_gb,
                                                        num_examples_per_epoch=10000,
                                                        )
            self.train_dataset = CombinedDataset((self.train_dataset_1, 
                                                  self.train_dataset_2))
            

        if stage == "validate" or stage == "fit":
            self.train_dataset_1 = self.train_dataset_1(root_dirs= self.hparams.root_dirs, 
                                                        indices = [150,200], 
                                                        min_tracks =self.hparams.min_tracks, 
                                                        max_tracks = self.hparams.max_tracks, 
                                                        length = self.hparams.length, 
                                                        buffer_size_gb=self.hparams.val_buffer_size_gb,
                                                        num_examples_per_epoch=1000,
                                                        )
            self.train_dataset_2 = self.train_dataset_2(self.hparams.root_dirs, 
                                                        subset = "val", 
                                                        min_tracks = self.hparams.min_tracks, 
                                                        max_tracks = self.hparams.max_tracks, 
                                                        length = self.hparams.length, 
                                                        buffer_size_gb=self.hparams.val_buffer_size_gb,
                                                        num_examples_per_epoch=1000,
                                                        )
            self.val_dataset = CombinedDataset((self.train_dataset_1, 
                                                self.train_dataset_2))
        
        if stage == "validate" or stage == "fit":
            self.train_dataset_1 = self.train_dataset_1(self.hparams.root_dirs, 
                                                        indices = [150,200], 
                                                        min_tracks = self.hparams.min_tracks, 
                                                        max_tracks = self.hparams.max_tracks, 
                                                        length = self.hparams.length, 
                                                        buffer_size_gb=self.hparams.val_buffer_size_gb,
                                                        num_examples_per_epoch=1000,
                                                        )
            self.train_dataset_2 = self.train_dataset_2(self.hparams.root_dirs, 
                                                        subset = "test", 
                                                        min_tracks = self.hparams.min_tracks, 
                                                        max_tracks = self.hparams.max_tracks, 
                                                        length = self.hparams.length, 
                                                        buffer_size_gb=self.hparams.val_buffer_size_gb,
                                                        num_examples_per_epoch=1000,
                                                        )
            self.test_dataset = CombinedDataset((self.train_dataset_1, self.train_dataset_2))
            

    def train_dataloader(self):
        # dataset =CombinedDataset(CambridgeDataset(root_dirs=["/import/c4dm-multitrack-private/C4DM Multitrack Collection/mixing-secrets"]),
        #                      MedleyDBDataset(["/import/c4dm-datasets/MedleyDB_V1/V1", "/import/c4dm-datasets/MedleyDB_V2/V2"]))
        loader = torch.utils.data.DataLoader(self.train_dataset,
                                             batch_size=self.hparams.batch_size, 
                                             shuffle=True, 
                                             drop_last = True, 
                                             num_workers=self.hparams.num_workers)
        return loader
    
    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.val_dataset,
                                             batch_size=self.hparams.batch_size, 
                                             shuffle=True, 
                                             drop_last = True, 
                                             num_workers=self.hparams.num_workers)
        return loader
    
    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(self.test_dataset,
                                             batch_size=self.hparams.batch_size, 
                                             shuffle=True, 
                                             drop_last = True, 
                                             num_workers=self.hparams.num_workers)
        return loader
    
        
    